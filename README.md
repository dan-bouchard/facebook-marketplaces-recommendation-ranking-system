# Facebook Marketplace Recommendation Ranking System

> A project to develop and train a Facebook Marketplace Search Ranking system which uses is a trained multimodal model accepting both text and image data in order to generate vector embeddings in order to make recommendations for a user searching for a product to buy.

## Cleaning the dataset

Cleaning the tabular data which has been scrapped from Facebook, with information regarding:
 - Product Name
 - Product Description
 - Product Category
 - Product Price
 - Location of the seller

 Also, cleaning the images by converting them all to 3-channel RGB and size 512x512.

**Original**                                              | **Cleaned**
----------------------------------------------------------|-------------------------------------------------------
![](./00a13f7d-b1ef-4754-8a99-3bebdf4604bb_original.jpg)  |![](./00a13f7d-b1ef-4754-8a99-3bebdf4604bb_cleaned.jpg)

**Original**                                              | **Cleaned**
----------------------------------------------------------|-------------------------------------------------------
<img src="./00a13f7d-b1ef-4754-8a99-3bebdf4604bb_original.jpg" width="400" />  |<img src="./00a13f7d-b1ef-4754-8a99-3bebdf4604bb_cleaned.jpg" width="400" />

 <p float="left">
  <img src="./00a13f7d-b1ef-4754-8a99-3bebdf4604bb_original.jpg" width="100" />
  <img src="./00a13f7d-b1ef-4754-8a99-3bebdf4604bb_cleaned.jpg" width="100" /> 
</p>
