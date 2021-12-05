# CosineAnnealingWithWarmup

## Formulation
The learning rate is annealed using a cosine schedule over the course of learning of _n\_total_ total steps with an initial warmup period of _n\_warmup_ steps. Hence, the learning rate at step _i_ is computed as:
![image](https://user-images.githubusercontent.com/37068560/144735206-ae709166-9fc0-4e32-9f54-868505a5cc67.png)

## Usage
```python
# optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch
lr_scheduler = LR_Scheduler(
        optimizer,
        args.warmup_epochs, args.warmup_lr*args.batch_size/256, 
        args.epochs, args.lr*args.batch_size/256, args.final_lr*args.batch_size/256, 
        len(train_loader),
    )

for data in range(train_loader):
  optimizer.zero_grad()
  loss = model(data) 
  loss.backward()
  optimizer.step()
  lr_scheduler.step()
```
