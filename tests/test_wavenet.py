import time
from model import WaveNetModel, Optimizer, WaveNetData

model = WaveNetModel(num_layers=10,
					 num_blocks=2,
					 num_classes=64,
					 hidden_channels=32)

print("model: ", model)
print("scope: ", model.scope)

data = WaveNetData('../train_samples/saber.wav',
				   input_length=model.scope,
				   target_length=model.last_block_scope,
				   num_classes=model.num_classes)
start_tensor = data.get_minibatch([0])[0].squeeze()

optimizer = Optimizer(model,
                      learning_rate=0.001,
					  max_steps=40,
                      stop_threshold=0.1,
                      avg_length=10)

print('start training...')
tic = time.time()
optimizer.train(data)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

print('generate...')
tic = time.time()
generated = model.generate(start_data=start_tensor, num_generate=100)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))

print('generate...')
tic = time.time()
generated = model.fast_generate(100)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))



