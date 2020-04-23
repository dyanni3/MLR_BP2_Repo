print('hello import')

from mlpluginapi import MLPluginAPI
import unreal_engine as ue
import json
import torch
import numpy as np

#MLPluginAPI
class myAPI(MLPluginAPI):

	#optional api: setup your model for training
	def on_setup(self):
		print('hello on_setup')
		print('the blueprint version is running!')
		ue.log('hi')
		pass
		
	#optional api: parse input object and return a result object, which will be converted to json for UE4
	def on_json_input(self, input_):
		print('hello on_json_input')
		print(f"python side: {input_}")
		x_loc = input_['stateValues']
		print(f"Current State: {x_loc}")
		tt = torch.tensor(-np.array(x_loc))
		#ret_val = torch.exp(tt)
		ret_val = {"Name":"Current Action", "ActionValues":[tt.tolist()[0]]}
		ret_val = json.dumps(ret_val)
		print(ret_val)
		return (ret_val)
		#return({"ret ret ret ret": "floof floof"})

	#optional api: start training your network
	def on_begin_training(self):
		#print('hello on_begin_training')
		pass

	def on_float_array_input(self, float_array_input):
		print("got a float array")
		ue.log(float_array_input)
		return(9.0)

	def on_test_string(self, string_input):
		print(f"python side: {input}")
		return ('{"b":"c"}')


#NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
#required function to get our api
def get_api():
	#return CLASSNAME.get_instance()
	return myAPI.get_instance()