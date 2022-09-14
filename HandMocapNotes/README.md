## Overview

![HandMocapNotes](./HandMocapNotes.png)


## Hand Estimate

### Param Model

- MANO

	- SMPL+H
	- 61 （3cam+3rot+15*3+10shape）

### MANO params

- minimal-hand

	- novel points

		- DetNet + IKNet

			- 100fps

	- pipe

		- image -> 2d+3d detect -> mano shape IK -> hand mesh

	- dataset

		- DetNet

			- CMU ( CMU panoptic dataset
			- RHD ( rendered hand pose dataset
			- GAN (GANerated Hands Dataset

		- IKNet

			- MANO mocap data + interpolation aug

- MobileHand

	- novel points

		- mobilenetv3

			- 110 Hz on a GPU or 75 Hz on a CPU

		- 23 degrees of freedom

			- replace MANO&PCA

	- pipe

		- image -> camera/mano param -> project 2d to cal loss

	- dataset

		- train&val

			- FreiHand/STB

	- metrics

		- 3d kps

			- PCK/AUC

		- hand shape

			- mesh error/F-score

- S2HAND

	- novel points

		- self supervised
		- 2D-3D Consistency Loss

	- dataset

		- FreiHand/HO3D

### SMPLX params

- FrankHand

	- novel points

		- encoder(r50)-decoder HMR like network

	- pipe

		- image -> image feature -> smplx param -> smplx mesh

	- dataset

		- train

			- FreiHand/HO-3D/MTC/STB/RHD/MPII+NZSL

		- test

			- STB/RHD/MPII+NZSL

### MANO mesh

- MobRecon

	- novel points

		- lightweight
		- SpiralConv

	- pipe

		- image -> 2d detect -> 3d lifting -> Regress Mesh (SpiralConv)

	- dataset

		- FreiHand/Human3.6M
		- self

			- real world testset
			- complement data

		- test

			- FreiHand/RHD/HO3Dv2

	- metrics

		- MPJPE/PA-MAPJPE/Acc/AUC/F-Score

- HandOccNet

	- novel points

		- FPN+FIT(feature injecting transformer)+SET(self-enhancing transformer)

	- dataset

		- HO3D/FPHA(first-person hand action)
