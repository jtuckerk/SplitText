# Script to make multiprocessing slightly easier


import multiprocessing as mp
import os

def GetCPUs():
  return mp.cpu_count()

def Bell():
  duration = 1  # seconds
  freq = 440  # Hz
  os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

# this is fast but probably doesn't do that well with large amounts of memory
def RunInParallel(arg_list, worker_fn, serially=False):
  """run worker over args in parallel
  arg_list: list of objects to be processed.
  worker_fn: any func that takes a single arg and returns a result.
  serially: for testing - easier to debug.
  """ 

  # debugging with dry_run=true is easier.
  if serially:
    outs = []
    for argv in arg_list:
      outs.append(worker_fn(argv))
    return outs
    
  def runner(in_queue, out_queue):
    for argv in iter(in_queue.get, "STOP"):
      out_queue.put(worker_fn(argv))

  in_queue = mp.Queue()
  out_queue = mp.Queue()

  # setup workers
  numProc = 10
  process = [mp.Process(target=runner,
                        args=(in_queue, out_queue)) for x in range(numProc)]

  # run processes

  # iterator over rows

  # fill queue and get data
  # code fills the queue until a new element is available in the output
  # fill blocks if no slot is available in the in_queue
  outs = []

  for argv in arg_list:
    try:
      in_queue.put(argv, block=True)  # row = (index, A, B, C, D) tuple
    except StopIteration:
      break

  for p in process:
    p.start()

  for i in range(len(arg_list)):
    row_data = out_queue.get()
    outs.append(row_data)
  # signals for processes stop

  for p in process:
    in_queue.put('STOP')

  # wait for processes to finish
  for p in process:
    p.join()
  Bell()
  return outs
