# Overview

 ![AIArt](./AIArt.png)

# AI Art

## Talking Head

### HumanFace

- Image Animation

	- monkey-net

		- network

			- unsupervised kp detector
			- dense motion netwrok
			- motion transfer network

	- FOM

		- metrics

			- same as monkey-net

		- network

			-  unsupervised keypoint detector

				- jacobian

			-  local affine transformations
			-  dense motion network

				- transform map
				- occlusion map

		- datasets

			- Thai-Chi-HD

- DaGAN

	- loss

		- PSNR/SSIM/AKD/AED
		- same as MarioNETte

	- dataset

		- VoxCeleb

	- network

		- Depth-Aware

			- face depth model

		- Depth-guided Facial Keypoints Detection
		- cross-modal attention machanism

			- face kp
			- face depth

- vid2vid

- TPSMM

### Anime

- TalkingHeadAnime

	- pose(6 -> face

		- left/right eye, mouth, head x/y/z

	- data

		- collect from MMD

	- network

		- face morpher

			- GANimation based

				- 2 generator like cyclegan
				- alpha mask + alpha blend
( attention based generator

		- face rotator

			- EnhancedView based

- TalkingHeadAnime2

	- update

		- manual pose
		- more expressive (eyebrow
		- ifacialmocap

	- network

		- pose 6 -> 42
		- eyebrow morpher

			- segment + remove + change

		- eye&mouth morpher

- EasyVtuber

	- mediapipe -> face mesh -> vector -> THA
	- obs stream

- EasyVtuber2

	- anime source

		- waifu labs

		- crypko.ai

	- ifacialmocap + UDP
	- obs -> unity capture

### Metrics

- PSNR
- SSIM
- from monkey-net

	-  AKD (Average Keypoint Distance
	-  AED (Average Euclidean Distance
	- MKR (Missing Keypoint Rate
	- for video

		- L1
		- FID (Frechet Inception Distance

- from MarioNETte

	- CSIM

		-  Cosine similarity for identity

	- PRMSE

		-  root mean square error of the head pose angles for head pose

	- AUCON

		-  the ratio of identical facial action unit values

## Text 2 Video

### Make A Video (meta

- pipe

	- text(CLIP to image Prior
	- spatiotemporal conv decoder
	- frame interpolation
	- SR+spatiotemporal SR

- datasets

	- LAION-5B
	- WebVid-10M
	- HD-VILA-100M

- metrics

	- FVD (Frechet Video Distance
	- FID (Frechet Inception Distance
	- CLIPSIM (CLIP similarity between video frames

### Imagen Video (google

- pipe

	- 7 models
	- text-conditional video genereate
	- spatial&temporal SR

- datasets

	- 14m internet video-text pair
	- 60m image-text pair
	- LAION-400M

### Phenaki (google

### CogVideo

- pipe

	- multi-frame-rate hierarchical training

		- frame rate/text/frame token
		- stage1 sequential gen
		- stage2 recursive interpolation

	- dual-channel attention

		- freeze CogView2
		- add sptial-temporal attention channel

## Text to Image

### service

- image generate service

	- midjourney

	- pornpen

	- novel ai

		- aitags

	- chinese

		- yige

- local/colab/hugging face

	- multimodalart

		- mindseye

			- using colab to run different models

		- majesty-diffusion

			- Latent Diffusion
			- V-Objective Diffusion

	- DreamBooth

		- Dreambooth-Stable-Diffusion

		- fast-stable-diffusion

			- fully colab

	- NovelAI

		- tag generate

### Diffusion

- DALL-E
- GLIDE

- Latent Diffusion (CompVis

	- diffusion in latent space
	- Stable Diffusion

		- pipe

			-  860M UNet
			- CLIP ViT-L/14 text encoder

		- datasets

			- LAION-5B

- disco-diffusion

- ERNIE-ViLG

- Imagen (Google

	- Dreambooth

		- use few images finetune T2I model

### Transformer

- CogView2

## Image Generate

### NFT

- generate

	- generate & interpolation & style transfer

	- image compose

- website

	- nftcn

	- opensea

### AnimeFace

- StyleGan

	- NVlabs
	- stylegan2-ada-pytorch

		- upfirdn2d

- Competitor

	- Anime

		- crypko.ai

		- waifu

### Semantic Image Synthesis

- SPADE

	- spatially-adaptive normalization

		- semantic mask -> scale/bias

	- generator

		- pix2pixHD remove encoder

## Nerf

