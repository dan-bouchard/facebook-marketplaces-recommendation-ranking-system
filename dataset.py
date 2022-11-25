import numpy as np
import pandas as pd
import random
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision.transforms import ToTensor
import torch


def set_seed(seed):
    '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    # os.environ['PYTHONHASHSEED'] = str(SEED)


def import_tabular_data():
    products_df = pd.read_csv('Products.csv', lineterminator='\n')
    cleaned_df = clean_tabular_data(products_df)
    images_df = pd.read_csv('Images.csv', lineterminator="\n").drop(columns='Unnamed: 0')

    merged_df = (pd.merge(images_df, cleaned_df, left_on='product_id', right_on='id', how='inner', validate='many_to_one', suffixes=('', '_y'))
                    .drop(columns='id_y')
                    .rename(columns= lambda col: 'image_id' if col == 'id' else col))
    return merged_df


def clean_tabular_data(df):
    return (df.drop(columns='Unnamed: 0')
                .assign(price = df.price.str.replace(',',''))
                .assign(price = lambda df_: df_.price.str.extract(r'£([\d]+.[\d]+)').astype('float'),
                        category_0 = df.category.str.split(' / ', expand=True)[0],
                        category_1 = df.category.str.split(' / ', expand=True)[1])
                .drop(columns='category')
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_details, img_dir, encoder):
        self.img_id = img_details.image_id
        self.img_labels = img_details.category_0
        self.img_labels_encoded = self.img_labels.replace(encoder)
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.img_id)
    
    def __getitem__(self, idx):
        img_path = img_path = self.img_dir + self.img_id.iloc[idx] + '.jpg'
        transform = ToTensor()
        image = transform(Image.open(img_path))
        label = torch.tensor(self.img_labels_encoded.iloc[idx]).type(torch.uint8)
        return image, label


def generate_dataloaders(seed=42, batch_size = 64):
    set_seed(seed)
    merged_df = import_tabular_data()
    labels = sorted(merged_df.category_0.unique())
    encoder = dict(zip(labels, range(len(labels))))
    decoder = dict(zip(range(len(labels)), labels))

    training_idx, test_idx = train_test_split(np.arange(merged_df.shape[0]), test_size=0.15, shuffle=True, stratify=merged_df.category_0)
    train_idx, valid_idx = train_test_split(training_idx, test_size=0.15/0.85, shuffle=True, stratify=merged_df.loc[training_idx].category_0)

    train_dataset = Dataset(merged_df.loc[train_idx], './cleaned_images_64/', encoder)
    val_dataset = Dataset(merged_df.loc[valid_idx], './cleaned_images_64/', encoder)
    test_dataset = Dataset(merged_df.loc[test_idx], './cleaned_images_64/', encoder)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {'train': train_dataloader, 
                    'validation': val_dataloader,
                    'test': test_dataloader}
    return dataloaders
