# CosineAnnealingWithWarmup

## Formulation
The learning rate is annealed using a cosine schedule over the course of learning of _n\_total_ total steps with an initial warmup period of _n\_warmup_ steps. Hence, the learning rate at step _i_ is computed as:<br>
<img src="https://user-images.githubusercontent.com/37068560/144736381-735a21cf-476e-4728-a785-cbe59858b24e.png" width="600px" />


Learning rate will be changed as:<br>
<img src="https://user-images.githubusercontent.com/37068560/144736137-5bb1eeb5-70bf-4e18-8302-3937a718610b.png" width="300px" />


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
  output = model(data) 
  loss = lossfunc(output,gt)
  loss.backward()
  optimizer.step()
  lr_scheduler.step()
```
In CV domain \[1,2\], in order to automatically adapt different batch size you can use a learning rate of lr×BatchSize/256 (linear scaling \[4\])(we can use larger learning rate while adopting larger batch size, especially, when you use LARS optimizer\[3\]). Of course, you can modify it according to your specific requirements.

## Reference
\[1\] Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Do- ersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Moham- mad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, R´emi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. arXiv:2006.07733v1, 2020. <br>
\[2\] Chen X, He K. Exploring simple siamese representation learning\[C\]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 15750-15758. <br>
\[3\] Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv:1708.03888, 2017. <br>
\[4\] Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noord- huis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677, 2017. <br>
\[5\] https://github.com/PatrickHua/SimSiam?utm_source=catalyzex.com <br>
