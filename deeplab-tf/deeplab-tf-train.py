
# suppress warnings from earlier versions of h5py (imported by tensorflow)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# don't write bytecode for imported modules to disk - different ranks can
#  collide on the file and/or make the file system angry
import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import argparse

# instead of scipy, use PIL directly to save images
try:
    import PIL
    def imsave(filename, data):
        PIL.Image.fromarray(data.astype(np.uint8)).save(filename)
    have_imsave = True
except ImportError:
    have_imsave = False

import h5py as h5
import os
import time

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#horovod, yes or no?
horovod=True
try:
    import horovod.tensorflow as hvd
except:
    horovod = False

#import helpers
try:
    script_path = os.path.dirname(sys.argv[0])
except:
    script_path = '.'
sys.path.append(os.path.join(script_path, '..', 'utils'))
from deeplab_model import *
from common_helpers import *
from data_helpers import *
import graph_flops

#import pycuda
try:
    import pycuda.autoinit
    import pycuda as pyc
    have_pycuda=True
    print("pycuda enabled")
except:
    print("pycuda not installed")
    have_pycuda=False

#GLOBAL CONSTANTS
image_height_orig = 768
image_width_orig = 1152


#load dictionary from argparse
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


#main function
def main(device, input_path_train, input_path_validation, dummy_data,
         downsampling_fact, downsampling_mode, channels, data_format,
         label_id, weights, image_dir, checkpoint_dir, trn_sz, val_sz,
         loss_type, model, decoder, fs_type, optimizer, batch, batchnorm, num_epochs, dtype,
         disable_checkpoints, disable_imsave, tracing, trace_dir, output_sampling, scale_factor,
         intra_threads, inter_threads):
    #init horovod
    comm_rank = 0
    comm_local_rank = 0
    comm_size = 1
    comm_local_size = 1
    if horovod:
        hvd.init()
        comm_rank = hvd.rank()
        comm_local_rank = hvd.local_rank()
        comm_size = hvd.size()
        #not all horovod versions have that implemented
        try:
            comm_local_size = hvd.local_size()
        except:
            comm_local_size = 1
        if comm_rank == 0:
            print("Using distributed computation with Horovod: {} total ranks".format(comm_size,comm_rank))

    #downsampling? recompute image dims
    image_height =  image_height_orig // downsampling_fact
    image_width = image_width_orig // downsampling_fact

    #parameters
    per_rank_output = False
    loss_print_interval = 1

    #session config
    sess_config=tf.ConfigProto(inter_op_parallelism_threads=inter_threads, #6
                               intra_op_parallelism_threads=intra_threads, #1
                               log_device_placement=False,
                               allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(comm_local_rank)
    sess_config.gpu_options.force_gpu_compatible = True

    #get data
    training_graph = tf.Graph()
    if comm_rank == 0:
        print("Loading data...")
    train_files = load_data(input_path_train, True, trn_sz, horovod)
    valid_files = load_data(input_path_validation, False, val_sz, horovod)

    #print some stats
    if comm_rank==0:
        print("Num workers: {}".format(comm_size))
        print("Local batch size: {}".format(batch))
        if dtype == tf.float32:
            print("Precision: {}".format("FP32"))
        else:
            print("Precision: {}".format("FP16"))
        print("Decoder: {}".format(decoder))
        print("Batch normalization: {}".format(batchnorm))
        print("Channels: {}".format(channels))
        print("Loss type: {}".format(loss_type))
        print("Loss weights: {}".format(weights))
        print("Loss scale factor: {}".format(scale_factor))
        print("Output sampling target: {}".format(output_sampling))
        #print optimizer parameters
        for k,v in optimizer.items():
            print("Solver Parameters: {k}: {v}".format(k=k,v=v))
        print("Num training samples: {}".format(train_files.shape[0]))
        print("Num validation samples: {}".format(valid_files.shape[0]))
        if dummy_data:
            print("Using synthetic dummy data")
        print("Disable checkpoints: {}".format(disable_checkpoints))
        print("Disable image save: {}".format(disable_imsave))

    #compute epochs and stuff:
    if fs_type == "local":
        num_samples = train_files.shape[0] // comm_local_size
    else:
        num_samples = train_files.shape[0] // comm_size
    print("num_samples: {} batch: {}".format(num_samples, batch))
    num_steps_per_epoch = num_samples // batch
    num_steps = num_epochs*num_steps_per_epoch
    if comm_rank==0:
        print("Number of steps per epoch: {}".format(num_steps_per_epoch))
        print("Number of steps in total: {}".format(num_steps))
    if per_rank_output:
        print("Rank {} does {} steps per epoch".format(comm_rank,
                                                       num_steps_per_epoch))

    with training_graph.as_default():

        if dummy_data:
            dummy_data_args = dict(batchsize=batch, data_format=data_format, dtype=dtype)
            trn_dataset = create_dummy_dataset(n_samples=trn_sz, num_epochs=num_epochs, **dummy_data_args)
            val_dataset = create_dummy_dataset(n_samples=val_sz, num_epochs=1, **dummy_data_args)
        else:
            #create readers
            trn_reader = h5_input_reader(input_path_train, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, data_format=data_format, label_id=label_id, sample_target=output_sampling)
            val_reader = h5_input_reader(input_path_validation, channels, weights, dtype, normalization_file="stats.h5", update_on_read=False, data_format=data_format, label_id=label_id)
            #create datasets
            if fs_type == "local":
                trn_dataset = create_dataset(trn_reader, train_files, batch, num_epochs, comm_local_size, comm_local_rank, dtype, shuffle=True)
                val_dataset = create_dataset(val_reader, valid_files, batch, 1, comm_local_size, comm_local_rank, dtype, shuffle=False)
            else:
                trn_dataset = create_dataset(trn_reader, train_files, batch, num_epochs, comm_size, comm_rank, dtype, shuffle=True)
                val_dataset = create_dataset(val_reader, valid_files, batch, 1, comm_size, comm_rank, dtype, shuffle=False)

        #create iterators
        handle = tf.placeholder(tf.string, shape=[], name="iterator-placeholder")
        iterator = tf.data.Iterator.from_string_handle(handle, (dtype, tf.int32, dtype, tf.string),
                                                       ((batch, len(channels), image_height_orig, image_width_orig) if data_format=="channels_first" else (batch, image_height_orig, image_width_orig, len(channels)),
                                                        (batch, image_height_orig, image_width_orig),
                                                        (batch, image_height_orig, image_width_orig),
                                                        (batch))
                                                       )
        next_elem = iterator.get_next()

        #if downsampling, do some preprocessing
        if downsampling_fact != 1:
            if downsampling_mode == "scale":
                #do downsampling
                rand_select = tf.cast(tf.one_hot(tf.random_uniform((batch, image_height, image_width), minval=0, maxval=downsampling_fact*downsampling_fact, dtype=tf.int32), depth=downsampling_fact*downsampling_fact, axis=-1), dtype=tf.int32)
                next_elem = (tf.layers.average_pooling2d(next_elem[0], downsampling_fact, downsampling_fact, 'valid', data_format), \
                             tf.reduce_max(tf.multiply(tf.image.extract_image_patches(tf.expand_dims(next_elem[1], axis=-1), \
                                                                                 [1, downsampling_fact, downsampling_fact, 1], \
                                                                                 [1, downsampling_fact, downsampling_fact, 1], \
                                                                                 [1,1,1,1], 'VALID'), rand_select), axis=-1), \
                             tf.squeeze(tf.layers.average_pooling2d(tf.expand_dims(next_elem[2], axis=-1), downsampling_fact, downsampling_fact, 'valid', "channels_last"), axis=-1), \
                             next_elem[3])
            elif downsampling_mode == "center-crop":
                #some parameters
                length = 1./float(downsampling_fact)
                offset = length/2.
                boxes = [[ offset, offset, offset+length, offset+length ]]*batch
                box_ind = list(range(0,batch))
                crop_size = [image_height, image_width]
                
                #be careful with data order
                if data_format=="channels_first":
                    next_elem[0] = tf.transpose(next_elem[0], perm=[0,2,3,1])
                    
                #crop
                next_elem = (tf.image.crop_and_resize(next_elem[0], boxes, box_ind, crop_size, method='bilinear', extrapolation_value=0, name="data_cropping"), \
                             ensure_type(tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(next_elem[1],axis=-1), boxes, box_ind, crop_size, method='nearest', extrapolation_value=0, name="label_cropping"), axis=-1), tf.int32), \
                             tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(next_elem[2],axis=-1), boxes, box_ind, crop_size, method='bilinear', extrapolation_value=0, name="weight_cropping"), axis=-1), \
                             next_elem[3])
                
                #be careful with data order
                if data_format=="channels_first":
                    next_elem[0] = tf.transpose(next_elem[0], perm=[0,3,1,2])
                    
            else:
                raise ValueError("Error, downsampling mode {} not supported. Supported are [center-crop, scale]".format(downsampling_mode))

        #create init handles
        #trn
        trn_iterator = trn_dataset.make_initializable_iterator()
        trn_handle_string = trn_iterator.string_handle()
        trn_init_op = iterator.make_initializer(trn_dataset)
        #val
        val_iterator = val_dataset.make_initializable_iterator()
        val_handle_string = val_iterator.string_handle()
        val_init_op = iterator.make_initializer(val_dataset)

        #compute the input filter number based on number of channels used
        num_channels = len(channels)
        #set up model
        model = deeplab_v3_plus_generator(num_classes=3, output_stride=8,
                                          base_architecture=model,
                                          decoder=decoder,
                                          batchnorm=batchnorm,
                                          pre_trained_model=None,
                                          batch_norm_decay=None,
                                          data_format=data_format)

        logit, prediction = model(next_elem[0], True, dtype)

        #set up loss
        loss = None

        #cast the logits to fp32
        logit = ensure_type(logit, tf.float32)
        if loss_type == "weighted":
            #cast weights to FP32
            w_cast = ensure_type(next_elem[2], tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1],
                                                          logits=logit,
                                                          weights=w_cast,
                                                          reduction=tf.losses.Reduction.SUM)
            if scale_factor != 1.0:
                loss *= scale_factor

        elif loss_type == "weighted_mean":
            #cast weights to FP32
            w_cast = ensure_type(next_elem[2], tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=next_elem[1],
                                                          logits=logit,
                                                          weights=w_cast,
                                                          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            if scale_factor != 1.0:
                loss *= scale_factor

        elif loss_type == "focal":
            #one-hot-encode
            labels_one_hot = tf.contrib.layers.one_hot_encoding(next_elem[1], 3)
            #cast to FP32
            labels_one_hot = ensure_type(labels_one_hot, tf.float32)
            loss = focal_loss(onehot_labels=labels_one_hot, logits=logit, alpha=1., gamma=2.)

        else:
            raise ValueError("Error, loss type {} not supported.",format(loss_type))

        #determine flops
        flops = graph_flops.graph_flops(format="NHWC" if data_format=="channels_last" else "NCHW", verbose=False, batch=batch, sess_config=sess_config)
        flops *= comm_size
        if comm_rank == 0:
            print('training flops: {:.3f} TF/step'.format(flops * 1e-12))

        #number of trainable parameters
        if comm_rank ==0:
            num_params = get_number_of_trainable_parameters()
            print('number of trainable parameters: {} ({} MB)'.format(num_params, num_params*(4 if dtype==tf.float32 else 2)*(2**-20)))
            
        if horovod:
            loss_avg = hvd.allreduce(ensure_type(loss, tf.float32))
        else:
            loss_avg = tf.identity(loss)

        #set up global step - keep on CPU
        with tf.device('/device:CPU:0'):
            global_step = tf.train.get_or_create_global_step()

        #set up optimizer
        if optimizer['opt_type'].startswith("LARC"):
            if comm_rank==0:
                print("Enabling LARC")
            train_op, lr = get_larc_optimizer(optimizer, loss, global_step,
                                              num_steps_per_epoch, horovod)
        else:
            train_op, lr = get_optimizer(optimizer, loss, global_step,
                                         num_steps_per_epoch, horovod)

        #set up streaming metrics
        iou_op, iou_update_op = tf.metrics.mean_iou(labels=next_elem[1],
                                                    predictions=tf.argmax(prediction, axis=3),
                                                    num_classes=3,
                                                    weights=None,
                                                    metrics_collections=None,
                                                    updates_collections=None,
                                                    name="iou_score")
        iou_reset_op = tf.variables_initializer([ i for i in tf.local_variables() if i.name.startswith('iou_score/') ])

        if horovod:
            iou_avg = hvd.allreduce(iou_op)
        else:
            iou_avg = tf.identity(iou_op)

        if "gpu" in device.lower():
            with tf.device(device):
                mem_usage_ops = [ tf.contrib.memory_stats.MaxBytesInUse(),
                                  tf.contrib.memory_stats.BytesLimit() ]
        #hooks
        #these hooks are essential. regularize the step hook by adding one additional step at the end
        hooks = [tf.train.StopAtStepHook(last_step=num_steps+1)]
        #bcast init for bcasting the model after start
        if horovod:
            init_bcast = hvd.broadcast_global_variables(0)
        #initializers:
        init_op =  tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()

        #checkpointing
        if comm_rank == 0:
            checkpoint_save_freq = 5*num_steps_per_epoch
            checkpoint_saver = tf.train.Saver(max_to_keep = 1000)
            if (not disable_checkpoints):
                hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=checkpoint_save_freq, saver=checkpoint_saver))
            #create image dir if not exists
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)

        #tracing
        if tracing is not None:
            import tracehook
            tracing_hook = tracehook.TraceHook(steps_to_trace=tracing,
                                               cache_traces=True,
                                               trace_dir=trace_dir)
            hooks.append(tracing_hook)
            print("############ tracing enabled")

        # instead of averaging losses over an entire epoch, use a moving
        #  window average
        recent_losses = []
        loss_window_size = 10
        #start session
        with tf.train.MonitoredTrainingSession(config=sess_config, hooks=hooks) as sess:
            #initialize
            sess.run([init_op, init_local_op])
            #restore from checkpoint:
            if comm_rank == 0 and not disable_checkpoints:
                load_model(sess, checkpoint_saver, checkpoint_dir)
            #broadcast loaded model variables
            if horovod:
                sess.run(init_bcast)
            #create iterator handles
            trn_handle, val_handle = sess.run([trn_handle_string, val_handle_string])
            #init iterators
            sess.run(trn_init_op, feed_dict={handle: trn_handle})
            sess.run(val_init_op, feed_dict={handle: val_handle})

            # figure out what step we're on (it won't be 0 if we are
            #  restoring from a checkpoint) so we can count from there
            train_steps = sess.run([global_step])[0]

            #do the training
            epoch = 1
            step = 1

            prev_mem_usage = 0
            t_sustained_start = time.time()
            r_peak = 0

            #warmup loops
            print("### Warmup for 5 steps")
            start_time = time.time()
            for _ in range(5):
                _ = sess.run([train_op],feed_dict={handle: trn_handle})
                #tmp_loss = sess.run([(loss if per_rank_output else loss_avg)],feed_dict={handle: trn_handle})
            end_time = time.time()
            print("### Warmup time: {:0.2f}".format(end_time - start_time))

            ### Start profiling
            print('Begin training loop')
            if have_pycuda:
                pyc.driver.start_profiler()

            while not sess.should_stop():
                try:
                    #construct feed dict
                    _ = sess.run([train_op],feed_dict={handle: trn_handle})
                    #tmp_loss = sess.run([(loss if per_rank_output else loss_avg)],feed_dict={handle: trn_handle})

                except tf.errors.OutOfRangeError:
                    break

            ### End of profiling
            if have_pycuda:
                pyc.driver.stop_profiler()

        # write any cached traces to disk
        if tracing is not None:
            tracing_hook.write_traces()

    print('All done')

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("--output",type=str,default='output',help="Defines the location and name of output directory")
    AP.add_argument("--chkpt_dir",type=str,default='checkpoint',help="Defines the location and name of the checkpoint file")
    AP.add_argument("--train_size",type=int,default=-1,help="How many samples do you want to use for training? A small number can be used to help debug/overfit")
    AP.add_argument("--validation_size",type=int,default=-1,help="How many samples do you want to use for validation?")
    AP.add_argument("--frequencies",default=[0.991,0.0266,0.13],type=float, nargs='*',help="Frequencies per class used for reweighting")
    AP.add_argument("--downsampling",default=1,type=int,help="Downsampling factor for image resolution reduction.")
    AP.add_argument("--downsampling_mode",default="scale",type=str,help="Which mode to use [scale, center-crop].")
    AP.add_argument("--loss",default="weighted",choices=["weighted","weighted_mean","focal"],type=str, help="Which loss type to use. Supports weighted, focal [weighted]")
    AP.add_argument("--datadir_train",type=str,help="Path to training data")
    AP.add_argument("--datadir_validation",type=str,help="Path to validation data")
    AP.add_argument("--dummy_data", action="store_true", help="Use synthetic data instead of real data")
    AP.add_argument("--channels",default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],type=int, nargs='*',help="Channels from input images fed to the network. List of numbers between 0 and 15")
    AP.add_argument("--fs",type=str,default="local",help="File system flag: global or local are allowed [local]")
    AP.add_argument("--optimizer",action=StoreDictKeyPair)
    AP.add_argument("--model",type=str,default="resnet_v2_101",help="Pick base model [resnet_v2_50, resnet_v2_101].")
    AP.add_argument("--decoder",type=str,default="bilinear",help="Pick decoder [bilinear,deconv,deconv1x]")
    AP.add_argument("--epochs",type=int,default=150,help="Number of epochs to train")
    AP.add_argument("--batch",type=int,default=1,help="Batch size")
    AP.add_argument("--use_batchnorm",action="store_true",help="Set flag to enable batchnorm")
    AP.add_argument("--dtype",type=str,default="float32",choices=["float32","float16"],help="Data type for network")
    AP.add_argument("--disable_checkpoints",action='store_true',help="Flag to disable checkpoint saving/loading")
    AP.add_argument("--disable_imsave",action='store_true',help="Flag to disable image saving")
    AP.add_argument("--disable_horovod",action='store_true',help="Flag to disable horovod")
    AP.add_argument("--tracing",default=None,type=str,help="Steps or range of steps to trace")
    AP.add_argument("--trace_dir",type=str,help="Directory where trace files should be written")
    AP.add_argument("--sampling",type=int,help="Target number of pixels from each class to sample")
    AP.add_argument("--scale_factor",default=0.1,type=float,help="Factor used to scale loss.")
    AP.add_argument("--device", default="/device:gpu:0",help="Which device to count the allocated memory on.")
    AP.add_argument("--data_format", default="channels_first",help="Which data format shall be picked [channels_first, channels_last].")
    AP.add_argument("--label_id", type=int, default=None,
                    help="Allows to select a certain label out of a multi-channel labeled data, \
                    where each channel presents a different label (e.g. for fuzzy labels). \
                    If set to None, the selection will be randomized [None].")
    # These are Thorsten's default; not sure they make sense
    AP.add_argument("--intra_threads", type=int, default=1, help='intra-op parallelism thread setting')
    AP.add_argument("--inter_threads", type=int, default=6, help='inter-op parallelism thread setting')
    parsed = AP.parse_args()

    #play with weighting
    weights = [1./x for x in parsed.frequencies]
    weights /= np.sum(weights)

    # convert name of datatype into TF type object
    dtype=getattr(tf, parsed.dtype)

    #check if we want horovod to be disabled
    if parsed.disable_horovod:
        horovod = False
        
    #invoke main function
    main(device=parsed.device,
         input_path_train=parsed.datadir_train,
         input_path_validation=parsed.datadir_validation,
         dummy_data=parsed.dummy_data,
         downsampling_fact=parsed.downsampling,
         downsampling_mode=parsed.downsampling_mode,
         channels=parsed.channels,
         data_format=parsed.data_format,
         label_id=parsed.label_id,
         weights=weights,
         image_dir=parsed.output,
         checkpoint_dir=parsed.chkpt_dir,
         trn_sz=parsed.train_size,
         val_sz=parsed.validation_size,
         loss_type=parsed.loss,
         model=parsed.model,
         decoder=parsed.decoder,
         fs_type=parsed.fs,
         optimizer=parsed.optimizer,
         num_epochs=parsed.epochs,
         batch=parsed.batch,
         batchnorm=parsed.use_batchnorm,
         dtype=dtype,
         disable_checkpoints=parsed.disable_checkpoints,
         disable_imsave=parsed.disable_imsave,
         tracing=parsed.tracing,
         trace_dir=parsed.trace_dir,
         output_sampling=parsed.sampling,
         scale_factor=parsed.scale_factor,
         intra_threads=parsed.intra_threads,
         inter_threads=parsed.inter_threads)
