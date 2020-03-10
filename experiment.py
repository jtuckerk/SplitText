import torch
import torch.nn as nn
from math import ceil
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import logging
import copy
import yaml
from collections import defaultdict

class Tracker():
  def __init__(self, console_metrics):
    self.tracked_types = defaultdict(lambda: defaultdict(list))
    self.console_metrics = console_metrics
    self.last_logged = {}
    for i in console_metrics:
      self.tracked_types[i] = defaultdict(list)

      self.last_logged[i] = 0
      self.extra = {} # column name: additional_space

    self.columns = ['Log Type']
    self.has_logged = False
    
  def Track(self, entry, item, value, add_space=None):
    self.tracked_types[entry][item].append(value)
    if add_space:
      self.extra[item] = add_space

  def _PopulateColumns(self):
    new_columns = False
    for t in self.console_metrics:
      type_dict = self.tracked_types[t]
      for metric_name in type_dict.keys():
        if metric_name not in self.columns:
          new_columns = True
          self.columns.append(metric_name)
    return new_columns

  def _LogSingle(self, log_type, format_str):
    values = [log_type.upper()]
    for v in self.columns[1:]:
      val = self.tracked_types[log_type].get(v, [""])[-1]
      val=str(val)[:len(v)+self.extra.get(v,0)]
      values.append(val)
    logging.info(format_str.format(*values))

  def Log(self, t):
    new_columns = self._PopulateColumns()
    s = ""
    for i, c in enumerate(self.columns):
      s +='{%d:<%d}' % (i, len(c)+2+self.extra.get(c,0))
    if new_columns or not self.has_logged:
      logging.info(s.format(*self.columns))

    self._LogSingle(t, s)
    self.has_logged = True
        
  def GetMetrics(self):
    return {t: dict(v) for t,v in self.tracked_types.items()}

Torch2Py = lambda x: x.cpu().numpy().tolist()

def Validate(val_loader, model, loss_fn, global_step, h, metric_tracker, val_type):
  cum_char_acc = 0
  cum_loss = 0
  cum_example_acc = 0

  for i, data in enumerate(val_loader):
    inputs = data['features']
    labels = data['labels']

    with torch.no_grad():
      outputs = model(inputs)
      cum_loss += loss_fn(outputs, labels)
      correct = ((outputs>.5)==labels)
      cum_char_acc += correct.float().mean()
      cum_example_acc += torch.min(correct, -1)[0].float().mean()

  steps = i+1
  metric_tracker.Track(val_type, 'global_step', global_step)
  metric_tracker.Track(val_type, 'char_accuracy', Torch2Py(cum_char_acc/steps))
  metric_tracker.Track(val_type, 'loss', Torch2Py(cum_loss.detach()/steps))
  metric_tracker.Track(val_type, 'example_accuracy', Torch2Py(cum_example_acc/steps))
  
def train(train_loader, train_d, validation_loader, model, loss_fn, optimizer, h, validate=True):

  n_batches = int(len(train_loader.dataset)/h['batch_size'])
  
  mt = Tracker(['train', 'val'])

  logging.info("TRAINING")
  tot_batches = n_batches*h['epochs']
  logging.info("total batches: %s" % tot_batches)
  global_step = 0

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=h['lr_step_size'], gamma=h['lr_decay'])
  bad_loss = False
  one_tenth = tot_batches // 10

  for epoch in range(h['epochs']):  # loop over the dataset multiple times
    start = time.time()
    running_loss = 0.0
    running_count = 0
    if bad_loss:
      break
    for i, data in enumerate(train_loader):
      global_step += 1
      inputs = data['features']
      labels = data['labels']

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
              
      loss.backward()
      optimizer.step()
      scheduler.step()
      # metrics
      running_loss += loss.item()
      running_count += 1

      if global_step % one_tenth == 0 or global_step < one_tenth and global_step % (one_tenth // 10) == 0:
        us_per_example = ((time.time()-start)/(h['batch_size']*(i+1)))*1e6
        correct = ((outputs>.5)==labels)
        char_acc = correct.float().mean()
        example_acc = torch.min(correct, -1)[0].float().mean()

        mt.Track('train', 'global_step', global_step)
        mt.Track('train', 'epoch', epoch)        
        mt.Track('train', 'char_accuracy', Torch2Py(char_acc))
        mt.Track('train', 'example_accuracy', Torch2Py(example_acc))        
        mt.Track('train', 'loss', Torch2Py(loss.detach()), 3)
        mt.Track('train', 'lr_at_step', scheduler.get_last_lr()[0])
        mt.Track('train', 'us/ex', us_per_example)
        mt.Track('train', '% cmplt', 100*global_step/tot_batches, -2)
        mt.Log('train')

      if loss.item() > 100 or torch.isnan(loss):
        bad_loss = True
        logging.info("Loss diverging. quitting")
        break        
      running_loss = running_count = 0.0

  if not bad_loss and h['run_validation']:
    Validate(validation_loader, model, loss_fn, global_step, h, mt, 'val')
    mt.Log('val')
  
  return mt.GetMetrics()
        
class DeviceDataset(torch.utils.data.Dataset):
  """Get a dataset from pt file directly to a device"""
  def __init__(self, torch_file, device='cuda', shuffle=False):
    self.data = torch.load(torch_file, map_location=device)
    self.keys=['features', 'labels']
    if shuffle:
      r = torch.randperm(self.nelement())
      for k in self.keys:
        self.data[k][r] = self.data[k]
    self.char_to_idx_map = self.data['char_to_idx_map']

  def __len__(self):
    return len(self.data[self.keys[0]])

  def __getitem__(self, idx):
    item = {
      'features': self.data['features'][idx].long(),
      'labels': self.data['labels'][idx].float()
      }
    return item

class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd
  def forward(self, x):
    return self.lambd(x)
      
class Net(nn.Module):
  def __init__(self, h):    
    super(Net, self).__init__()
    self.sequence_len = h['sequence_length']
    self.emb = nn.Embedding(h['vocab_size'], h['char_embedding_size'])

    if h['conv_activation'] == 'relu':
      conv_activation = nn.ReLU()
    elif h['conv_activation'] == 'sigmoid':
      conv_activation = nn.Sigmoid()

    self.convs = []

    last_out_chan = h['char_embedding_size']
    const_len_fn = lambda x: x[:,:,:self.sequence_len]
    for k, f in h['kernel|filter_sizes']:
      kern_size = k
      pad_size = ceil((kern_size-1)/2)
      self.convs.append(nn.Conv1d(last_out_chan, f, kern_size, padding=pad_size))
      self.convs.append(LambdaLayer(const_len_fn))
      self.convs.append(conv_activation)
      last_out_chan = f

    self.convs = nn.Sequential(*self.convs)

    kern_size = h['final_conv_kernel']
    pad_size = ceil((kern_size-1)/2)    
    self.conv_final = nn.Conv1d(last_out_chan, 1, kern_size, padding=pad_size)
    
  def forward(self, x):
    embeddings  = self.emb(x).permute(0,2,1)

    convout = self.convs(embeddings)
    logits = self.conv_final(convout)

    return nn.Sigmoid()(torch.flatten(logits, start_dim=1))

class NetWorks(nn.Module):
  def __init__(self, h):    
    super(NetWorks, self).__init__()
    assert len(h['filter_sizes']) == len(h['kernel_sizes']), "must have eq num of filters and kernels"
    self.sequence_len = h['sequence_length']
    self.emb = nn.Embedding(h['vocab_size'], h['char_embedding_size'])

    if h['activation'] == 'relu':
      self.activation = nn.ReLU()
    elif h['activation'] == 'sigmoid':
      self.activation = nn.Sigmoid()
      
    self.convs = []
    
    last_out_chan = h['char_embedding_size']
    for i, f in enumerate(h['filter_sizes']):
      kern_size = h['kernel_sizes'][i]
      pad_size = ceil((kern_size-1)/2)
      self.convs.append(nn.Conv1d(last_out_chan, f, kern_size, padding=pad_size))
      last_out_chan = f

    self.convs = nn.Sequential(*self.convs)  
    kern_size = h['final_conv_kernel']
    pad_size = ceil((kern_size-1)/2)    
    self.conv_final = nn.Conv1d(last_out_chan, 1, kern_size, padding=pad_size)

  def forward(self, x):
    embeddings  = self.emb(x).permute(0,2,1)
    activations = [embeddings]
    for conv in self.convs:
      conv_out = conv(activations[-1])[:,:,:self.sequence_len]
      conv_out = self.activation(conv_out)
      activations.append(conv_out)

    logits = self.conv_final(activations[-1])[:,:,:self.sequence_len]
    self.activations = activations
    activations.append(logits)
    return nn.Sigmoid()(torch.flatten(logits, start_dim=1))
  
def init_dataset(exp_info):
  ds = DeviceDataset(exp_info['dataset'], device='cuda')
  tr, va = exp_info['dataset_split'] # the rest is test
  n_train_examples, n_val_examples = int(len(ds)*tr), int(len(ds)*va)
  n_test_examples = len(ds)- n_train_examples - n_val_examples
  train_d, val_d, test_d = torch.utils.data.random_split(
    ds, [n_train_examples,n_val_examples,n_test_examples])
  logging.debug("n_train_examples: %s" % n_train_examples)
  logging.debug("n_val_examples: %s" % n_val_examples)

  return train_d, val_d, test_d,

def run_one(h, data):
  train_d, val_d, test_d  = data

  if 'seed' in h:
    torch.manual_seed(h['seed'])

  train_loader = DataLoader(train_d, batch_size=h['batch_size'], shuffle=True)
  validation_loader = DataLoader(val_d, batch_size=1000, shuffle=False)
  logging.debug("n_train_batches: %s" % (len(train_loader.dataset)//h['batch_size']))

  model = Net(h).cuda()
  logging.info("Experiment Model:\n" + str(model))
  logging.info("Hyperparams:\n" + str(h))

  size_params = 0
  size_bytes = 0

  for p in model.parameters():
    size_params += p.nelement()
    size_bytes += p.nelement()*p.element_size()
  logging.info("Model Size: {:.2e} bytes".format(size_bytes))

  if 'model_size_range_bytes' in h:
    min_size, max_size = h['model_size_range_bytes']
    if size_bytes < min_size or size_bytes > max_size:
      logging.info("Model size (%s bytes) outside of acceptable range. skipping" % size_bytes)
      return {'val': {'size_bytes': size_bytes,
                      'size_params': size_params},
              'exit_info': 'size outside of range. skipping'}, model

  if 'model_checkpoint' in h and h['model_checkpoint']:
    ckpt = h['model_checkpoint']
    r = model.load_state_dict(torch.load(ckpt))
    logging.info("Loaded model: %s" % str(r))
    
  optimizer = optim.SGD(model.parameters(), lr=h['learning_rate'], momentum=h['momentum'])

  loss_fn = nn.BCELoss()
    
  results = train(train_loader, train_d, validation_loader, 
                  model, loss_fn, optimizer, h, validate=True)

  results['val']['size_params'] = size_params
  results['val']['size_bytes'] = size_bytes

  return results, model
