import gdown

metadata_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Advanced%20Topics%20in%20AI/Sessions%206%20-%2010%20(Projects)/Project%20-%20Visual%20Question%20Answering/Annotations.zip'
metadata_path        = './metadata.zip'




gdown.download(metadata_url, './metadata.zip', True)
!unzip ./metadata.zip
data_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Advanced%20Topics%20in%20AI/Sessions%206%20-%2010%20(Projects)/Project%20-%20Visual%20Question%20Answering/small.h5'
image_data_path      = 'small.h5'




gdown.download(data_url, 'small.h5', True)
train_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Advanced%20Topics%20in%20AI/Sessions%206%20-%2010%20(Projects)/Project%20-%20Visual%20Question%20Answering/train.h5'
train_path      = 'train.h5'




gdown.download(train_url, 'train.h5', True)
val_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Advanced%20Topics%20in%20AI/Sessions%206%20-%2010%20(Projects)/Project%20-%20Visual%20Question%20Answering/val.h5'
val_path      = 'val.h5'





gdown.download(val_url, 'val.h5', True)
data_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Advanced%20Topics%20in%20AI/Sessions%206%20-%2010%20(Projects)/Project%20-%20Visual%20Question%20Answering/embedding_matrix.h5'
image_data_path      = 'embedding_matrix.h5'





gdown.download(data_url, 'embedding_matrix.h5', True)
embeddings_path = 'embedding_matrix.h5'
data_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/Deep%20Dives/Advanced%20Topics%20in%20AI/Sessions%206%20-%2010%20(Projects)/Project%20-%20Visual%20Question%20Answering/vocab_vizwiz%20(1).json'
image_data_path      = 'vocab.json'




gdown.download(data_url, 'vocab.json', True)
vocab_path = 'vocab.json'
