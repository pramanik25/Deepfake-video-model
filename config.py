import yaml
import os

#read yaml file
#Rreplaced model with New folder
#Rreplaced os.path.join('New folder','config.yaml') with a=os.path.join('New folder','config.yaml')
#Rreplaced  with open(a=os.path.join('New folder','config.yaml')) as file:
def load_config():
  with open(os.path.join('C:\\Users\\rajku\\OneDrive\\Desktop\\python folder\\model','C:\\Users\\rajku\\OneDrive\\Desktop\\python folder\\model\\config.yaml')) as file:
    config= yaml.safe_load(file)
  
  #q = type(config['model'])
  #print 
  #print(config['model'])
  #print(q)
  #model_from_call = config('model')
  #print("Type from callable:", type(model_from_call))
  #print("Value from callable:", model_from_call)

  #model_from_index = config['model']
  #print("Type from subscript:", type(model_from_index))
  #print("Value from subscript:", model_from_index)
  # If only the first element is needed:
  #if isinstance(model_from_call, list):
   # model_from_call = model_from_call[0]
  #print('model from index backbone',model_from_index['backbone'])
  #print('printing by get func',config.get("model", {}).get("backbone", {}))
  #print(config(['model'][0]['backbone']))
  return config
  print(config)

#def load_config():
#    config_path = os.path.join('model', 'config.yaml')
#    with open(config_path, 'r') as file:
#        config = yaml.safe_load(file)
    
#    if not config:
#        print("err1.1")
#        raise ValueError(f"The config file at {config_path} is empty or malformatted.")
       
#    if 'New folder' not in config:
 #       print("errr1.2")
        # raise KeyError(f"Missing key 'model' in the config file at {config_path}.")
        
#    return config
