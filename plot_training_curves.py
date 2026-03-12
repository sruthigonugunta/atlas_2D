import matplotlib.pyplot as plt

epochs = list(range(1,11))

train_loss = [0.0180,0.0062,0.0046,0.0038,0.0032,0.0028,0.0025,0.0023,0.0021,0.0020]
val_loss   = [0.0087,0.0094,0.0078,0.0072,0.0079,0.0083,0.0087,0.0091,0.0084,0.0090]

train_iou = [0.7453,0.7828,0.8046,0.8199,0.8311,0.8400,0.8482,0.8548,0.8620,0.8687]
val_iou   = [0.7642,0.7629,0.7785,0.7978,0.7962,0.8003,0.8016,0.7968,0.8027,0.8058]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs,train_loss,label="train loss")
plt.plot(epochs,val_loss,label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(epochs,train_iou,label="train IoU")
plt.plot(epochs,val_iou,label="val IoU")
plt.xlabel("epoch")
plt.ylabel("IoU")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("training_curves.png")

print("saved training_curves.png")
