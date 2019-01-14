### Evolution Strategies and Novelty Seeking Reward Evolution Strategies for training Generative Adversarial Networks 



Written report : [https://shareok.org/handle/11244/316799](https://shareok.org/handle/11244/316799) 

Code based on:
[Patryk Chrabaszcz et](https://github.com/PatrykChrabaszcz/Canonical_ES_Atari),
[Uber Neuroevolution](https://github.com/uber-research/deep-neuroevolution),
[OpenAI ES](https://github.com/openai/evolution-strategies-starter)


#### How to run : 

ES-GAN 
```
mpirun python3 main.py -e 1 -g ES_GAN -c configurations/es_gan_simple_configuration.json -r test_run -d datasets/
```

NS-ES-GAN/NSR-ES-GAN
```
mpirun python3 main.py -e 1 -g NS_GAN -c configurations/nsres_gan_simple_configuration.json -r test_run -d datasets/
```
Training using SGD : 
```
    python3 main_sgd.py -e 20 -g SGD_ADAM -c configurations/es_gan_simple_configuration.json -r test_run -d datasets/
```

*** on the 2d gan experiments, dataset folder is left empy
