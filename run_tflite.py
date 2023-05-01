import tflite_runtime.interpreter as tflite
import numpy as np
import time

interpreter = tflite.Interpreter("converted_model-int8.tflite")#, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

input_data = np.load("mnist_sample7.npy")
input_data = input_data.reshape(1, 28, 28, 1).astype('float32')
input_data /= 255.0
#print(input_data) # float32

#print(input_data)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

#print(output_details)

tensor_details = interpreter.get_tensor_details()
"""for detail in tensor_details:
    print(detail)
"""
st = time.perf_counter()
for _ in range(10000):
    interpreter.invoke()

en = time.perf_counter()

output_data = interpreter.get_tensor(output_details[0]['index'])
#output_data = interpreter.get_tensor(21)
print(output_data[0])
#print(output_data.transpose(0, 3, 1, 2)[0].shape)
#np.savetxt("my_np_" + 'Conv2D_17' + "_tf.txt", output_data.transpose(0, 3, 1, 2)[0][0], fmt = '%3.g')
print(en - st)

