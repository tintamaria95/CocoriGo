path = "/content/drive/MyDrive/GO_project/results_training/"
model_name = "baseline.h5"
f = open(path + model_name + ".txt", 'r')
lines =  f.readlines()
if (len(lines) != 0):
  iters = lines[0]
  mse = lines[1]
  policy_acc = lines[2]
else:
  iters = ''
  mse = ''
  policy_acc = ''

resRand = ['1', '6', '12', '34']
results = []
for r in resRand:
  results.append(r)
iters_str = ''
res_mse_str = ''

for i in range(len(resRand)):
  iters_str += (str(i) + ',')
  res_mse_str += (str(results[i]) + ',')
iters_str = iters_str[: -1] + '\n'
resStr = res_mse_str[: -1]  + '\n'

f.writelines(model_name + '\n')
f.writelines(iters_str)
f.writelines(res_mse_str)
