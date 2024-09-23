# Compact-public
Compact is scheme that produces piece-wise polynomial
approximations of complex activation functions (i.e., `silu`, `gelu`, `mish`). The generated approximation can be used
with state-of-the-art MPC techniques without the need to 
train the model  to retrain model accuracy. 
To achieve this, in compact, we infuse input density
awareness and use an application specific simulated annealing type
optimization to generate computationally efficient approximations
of complex activation functions. 

Compact is described in this paper: [**Compact: Approximating Complex Activation Functions for Secure Computation**](https://arxiv.org/pdf/2309.04664) accepted at Privacy Enhancing Technologies (PETs) 2024 conference.

## Results

## How to run 

## Acknowledgements  and Citations
Some part of the code is taken from the [NFGen](https://github.com/Fannxy/NFGen/tree/main).

If you use findings of this study in your research, please cite the following publication.
```
@article{compact,
  title={Compact: Approximating Complex Activation Functions for Secure Computation},
  author={Mazharul, Islam and Sunpreet, S. Arora, and Rahul, Chatterjee, and Peter, Rindal, and Maliheh, Shirvanian‡},
  journal={Proceedings on Privacy Enhancing Technologies},
  volume={2024},
  number={3},
  pages={25–41},
  year={2024}
}

```

## Todo
- [ ] Add the instructions to run the code
- [ ] Upload the trained model in Google Drive and share
- [ ] Add unit tests
- [ ] Use Docker


