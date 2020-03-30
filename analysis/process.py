#!/usr/bin/env python

import os,sys,glob

def process(fin):

  lTime  = []
  lFlop  = []
  lL1    = []
  lL2    = []
  lDram  = []

  res = open(fin,"r")
  prevLine = ""

  for line in res:

    ### Time
    if ('smsp__cycles_elapsed.sum.per_second' in line) :
      linesp = line.split(',')
      tmpRate = float(linesp[len(linesp)-1].strip('\n').strip('"'))
      linesp = prevLine.split(',')
      tmpTotal = float(linesp[len(linesp)-1].strip('\n').strip('"'))
      lTime.append(tmpTotal/tmpRate)

    ### SP FLOP
    # add + mul
    if ('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum' in line) :
      linesp=line.split(',')
      lFlop.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum' in line) :
      linesp=line.split(',')
      lFlop.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    # fma
    if ('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum' in line):
      linesp=line.split(',')
      lFlop.append(float(linesp[len(linesp)-1].strip('\n').strip('"')) * 2.)

    ### L1 transactions
    # global
    if ('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    # local
    if ('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    # shared
    if ('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    # atomic
    if ('l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    if ('l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum' in line) :
      linesp=line.split(',')
      lL1.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))

    ### L2 transactions
    # read and write
    if ('lts__t_sectors_op_read.sum' in line) or ('lts__t_sectors_op_write.sum' in line) :
      linesp=line.split(',')
      lL2.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))
    # atomic
    if ('lts__t_sectors_op_red.sum' in line) or ('lts__t_sectors_op_atom.sum' in line) :
      linesp=line.split(',')
      lL2.append(float(linesp[len(linesp)-1].strip('\n').strip('"')) * 2.)

    ### DRAM transactions
    if ('dram__sectors_write.sum' in line) or ('dram__sectors_read.sum' in line):
      linesp=line.split(',')
      lDram.append(float(linesp[len(linesp)-1].strip('\n').strip('"')))

    prevLine = line     # end of for

  res.close()

  transactionSize = 32.
  GIGA = 1024. * 1024. * 1024.

  time      = sum(lTime)
  flop      = sum(lFlop)
  bytesL1   = sum(lL1)   * transactionSize
  bytesL2   = sum(lL2)   * transactionSize
  bytesDram = sum(lDram) * transactionSize

  print(time)
  print(flop)

if __name__== "__main__":
  # Get all profiling files
  files = []
  files += [ f for f in os.listdir('.') if f.startswith("metrics.")]
  files.sort()
  if not files:
    raise RuntimeError("No profiling data found")

  # Process profiling data
  for f in files:
      process(f)
