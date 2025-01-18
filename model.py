from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        # Squeeze: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get the batch size and number of channels
        batch_size, num_channels, _, _ = x.size()

        # Squeeze operation
        y = self.global_avg_pool(x).view(batch_size, num_channels)

        # Excitation operation
        y = self.fc(y).view(batch_size, num_channels, 1, 1)

        # Scale the input tensor by the excitation weights
        return x * y.expand_as(x)
    

class SelfAttention(nn.Module):
  def __init__(self,base_channel):
    super(SelfAttention, self).__init__()
    self.x_f=nn.Conv2d(in_channels=base_channel,
                        out_channels=base_channel // 8,
                        kernel_size=1,
                        stride=1,
                        padding='same')
    self.x_g=nn.Conv2d(in_channels=base_channel,
                        out_channels=base_channel // 8,
                        kernel_size=1,
                        stride=1,
                        padding='same')
    self.x_h=nn.Conv2d(in_channels=base_channel,
                        out_channels=base_channel ,
                        kernel_size=1,
                        stride=1,
                        padding='same')

    self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    self.softmax  = nn.Softmax(dim=-1)

  def forward(self,x):
      batch,C,H,W=x.size()
      x_f=self.x_f(x).view(batch,-1,H*W).permute(0,2,1)
      x_g=self.x_g(x).view(batch,-1,H*W)
      x_h=self.x_h(x).view(batch,-1,W*H)
      s=torch.bmm(x_f,x_g)
      attention=self.softmax(s)
      out = torch.bmm(x_h,attention.permute(0,2,1) )
      out = out.view(batch,C,W,H)

      out = self.gamma*out + x
      return out,attention


class RoiPoolingConv(nn.Module):
    def __init__(self, pool_size, num_rois, rois_mat):
        super(RoiPoolingConv, self).__init__()
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.rois = rois_mat

    def forward(self, x):
        # x is expected to be a 4D tensor: (batch_size, channels, height, width)
        input_shape = x.shape

        outputs = []

        for roi_idx in range(self.num_rois):
            # Extract ROI coordinates
            x_roi = self.rois[roi_idx, 0]
            y_roi = self.rois[roi_idx, 1]
            w_roi = self.rois[roi_idx, 2]
            h_roi = self.rois[roi_idx, 3]

            # Calculate lengths for pooling regions
            row_length = w_roi / float(self.pool_size)
            col_length = h_roi / float(self.pool_size)

            # Pooling regions
            pooled_regions = []
            for jy in range(self.pool_size):
                for ix in range(self.pool_size):
                    # Calculate coordinates for each pooling region
                    x1 = int(x_roi + ix * row_length)
                    x2 = int(x1 + row_length)
                    y1 = int(y_roi + jy * col_length)
                    y2 = int(y1 + col_length)

                    # Ensure coordinates are within bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(input_shape[3], x2)  # Width limit
                    y2 = min(input_shape[2], y2)  # Height limit

                    # Crop the image and perform max pooling
                    x_crop = x[:, :, y1:y2, x1:x2]
                    pooled_val = F.adaptive_max_pool2d(x_crop, (1, 1))  # Pool to (1, 1)
                    pooled_regions.append(pooled_val)

            # Concatenate pooled regions for this ROI
            pooled_regions_tensor = torch.cat(pooled_regions, dim=0)
            outputs.append(pooled_regions_tensor)

        # Stack outputs and reshape to final output shape
        final_output = torch.stack(outputs)  # Shape: (num_rois, batch_size, channels, 1, 1)
        final_output = final_output.view(-1, self.num_rois, self.pool_size, self.pool_size, input_shape[1])  # Reshape

        return final_output.permute(0, 1, 4, 2, 3)  # Rearrange
    


class stack(nn.Module):
    def __init__(self, num_rois, feat_dim):
        super(stack, self).__init__()
        self.num_rois=num_rois
        self.feat_dim=feat_dim

    def crop(self, dimension, start, end, x):
        """Crops (or slices) a Tensor on a given dimension from start to end."""
        if dimension == 0:
            return x[start:end]
        if dimension == 1:
            return x[:, start:end]
        if dimension == 2:
            return x[:, :, start:end]
        if dimension == 3:
            return x[:, :, :, start:end]
        if dimension == 4:
            return x[:, :, :, :, start:end]
        else:
            raise ValueError("Dimension out of range")

    def process_rois(self, roi_pool, num_rois, feat_dim, x_final):
        """Processes ROIs by cropping and reshaping."""
        jcvs = []

        for j in range(num_rois):
            # Crop the roi_pool tensor for the current ROI
            roi_crop = self.crop(1, j, j + 1, roi_pool)  # Assuming roi_pool has shape (batch_size, num_rois, features)

            # Apply squeeze function (similar to Lambda in Keras)
            x = roi_crop.squeeze(1)  # Remove dimension 1 (num_rois)

            # Reshape to (batch_size, feat_dim)
            x = x.contiguous().view(-1, feat_dim)

            jcvs.append(x)

        # Process final output
        x_final_reshaped = x_final.contiguous().view(-1, feat_dim)  # Reshape x_final

        jcvs.append(x_final_reshaped)

        # Stack all processed tensors along dimension 0 (batch dimension)
        stacked_output = torch.stack(jcvs, dim=1)

        return stacked_output

    def forward(self, roi_pool, x_final):
        """Forward pass through the CAP model."""
        return self.process_rois(roi_pool, self.num_rois, self.feat_dim, x_final)
    

class SeqSelfAttention(nn.Module):
    def __init__(self, units=27,feature_dim:int=None,attention_activation=None, return_attention=False):
        super(SeqSelfAttention, self).__init__()
        self.units = units
        self.attention_activation=attention_activation
        self.return_attention = return_attention
        self.feature_din=feature_dim
        # Weight matrices
        self.Wt = nn.Parameter(torch.Tensor(units,feature_dim ))
        self.Wx = nn.Parameter(torch.Tensor(units,feature_dim))
        self.bh = nn.Parameter(torch.Tensor(units))
        self.Wa = nn.Parameter(torch.Tensor(units, 1))
        self.ba = nn.Parameter(torch.Tensor(1))

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.Wt)
        nn.init.xavier_normal_(self.Wx)
        nn.init.zeros_(self.bh)
        nn.init.xavier_normal_(self.Wa)
        nn.init.zeros_(self.ba)

    def forward(self, inputs):
        batch_size, input_len, _ = inputs.size()

        # Compute attention scores
        q = inputs @ self.Wt.T  # (batch_size, input_len, units)
        k = inputs @ self.Wx.T  # (batch_size, input_len, units)

        beta = torch.tanh(q.unsqueeze(2) + k.unsqueeze(1) + self.bh)  # (batch_size, input_len, input_len, units)

        alpha = (beta @ self.Wa).squeeze(-1) + self.ba  # (batch_size, input_len, input_len)


        alpha = self.attention_activation(alpha)


        # Softmax normalization
        alpha = F.softmax(alpha, dim=-1)

        # Compute context vector
        c_r = alpha @ inputs  # (batch_size, input_len, feature_dim)

        if self.return_attention:
            return c_r, alpha

        return c_r
    

class NetRVLAD(nn.Module):
    """Creates a NetRVLAD class (Residual-less NetVLAD)."""
    def __init__(self, feature_size, max_samples, cluster_size, output_dim):
        super(NetRVLAD, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size

        # Initialize weights
        self.cluster_weights = nn.Parameter(torch.randn(cluster_size,feature_size) * (1 / math.sqrt(feature_size)))
        self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * (1 / math.sqrt(feature_size)))
        self.Wn = nn.Parameter(torch.randn(output_dim,cluster_size * feature_size ) * (1 / math.sqrt(cluster_size)))

    def forward(self, reshaped_input):
        """Forward pass of a NetRVLAD block.

        Args:
            reshaped_input: Tensor of shape (batch_size*max_samples, feature_size)

        Returns:
            vlad: the pooled vector of size (batch_size, output_dim)
        """
        # Compute activation
        activation = F.linear(reshaped_input, self.cluster_weights, self.cluster_biases)

        # Apply softmax
        activation = F.softmax(activation, dim=-1)

        # Reshape activation to (batch_size, max_samples, cluster_size)
        activation = activation.view(-1, self.max_samples, self.cluster_size)

        # Transpose to (batch_size, cluster_size, max_samples)
        activation = activation.transpose(1, 2)

        # Reshape input to (batch_size, max_samples, feature_size)
        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)

        # Compute VLAD
        vlad = torch.bmm(activation, reshaped_input)  # Batch matrix multiplication
        vlad = vlad.transpose(1, 2)  # Transpose to (batch_size, feature_size, cluster_size)

        # L2 normalize vlad along the feature dimension
        vlad = F.normalize(vlad, p=2, dim=1)

        # Reshape to (batch_size, cluster_size * feature_size)
        vlad = vlad.contiguous().view(-1, self.cluster_size * self.feature_size)

        # Equation 3 in the paper: \hat{y} = W_N N_v
        vlad = F.linear(vlad, self.Wn)

        return vlad

    def output_shape(self):
        return (None, self.output_dim)
    

class Cap(nn.Module):
  def __init__(self,channels, pool_size,num_rois,rois_mat,feature_dim,hidden_size,cluster_size,out_dim, reduction_ratio=16,resolution=33,gridSize=3, minSize=1,):
     super(Cap, self).__init__()
     self.feature_dim=feature_dim
     self.channels=channels
     self.pool_size=pool_size
     self.num_rois=num_rois
     self.hidden_size=hidden_size
     self.feature_extractor=models.efficientnet_b0(weights="DEFAULT")
     self.feature_extractor=torch.nn.Sequential(*(list(self.feature_extractor.children())[:-2]))
     self.se_block = SEBlock(channels=channels)
     self.selfattention=SelfAttention(base_channel=channels)
     self.roipoolconv=RoiPoolingConv(pool_size=pool_size,num_rois=num_rois,rois_mat=rois_mat)
     self.stack=stack(num_rois=num_rois,feat_dim=feature_dim)
     self.seqselfattention=SeqSelfAttention(attention_activation=nn.Sigmoid(),feature_dim=feature_dim)
     self.lstm_layer = nn.LSTM(input_size=channels, hidden_size=hidden_size, batch_first=True)
     self.netrvlad=NetRVLAD(feature_size=hidden_size,max_samples=num_rois+1,cluster_size=cluster_size,output_dim=out_dim)
     self.batch_norm=nn.BatchNorm1d(out_dim)

  def forward(self,x):
    num_batch=x.shape[0]
    x=self.feature_extractor(x)
    x=self.se_block(x)
    x_final,att=self.selfattention(x)
    x_resized=F.interpolate(x_final, size=(42, 42), mode='bilinear', align_corners=False)
    roi_pool=self.roipoolconv(x_resized)
    stacked_pathes=self.stack(roi_pool,x_final)
    seq=self.seqselfattention(stacked_pathes)
    seq_reshaped = seq.view(num_batch, self.num_rois+1, self.channels, self.pool_size, self.pool_size)
    seq_pooled = F.adaptive_avg_pool2d(seq_reshaped.view(num_batch,self.num_rois+1, self.channels, self.pool_size, self.pool_size), (1, 1))
    seq_pooled=seq_pooled.view(num_batch,self.num_rois+1, self.channels)
    lstm_output, (h_n, c_n) = self.lstm_layer(seq_pooled)
    lstm_output=lstm_output.contiguous().view(-1,self.hidden_size)
    y_netrvlad=self.netrvlad(lstm_output)
    y=self.batch_norm(y_netrvlad)
    return y


def getROIS(resolution=33,gridSize=3, minSize=1):

	coordsList = []
	step = resolution / gridSize # width/height of one grid square

	#go through all combinations of coordinates
	for column1 in range(0, gridSize + 1):
		for column2 in range(0, gridSize + 1):
			for row1 in range(0, gridSize + 1):
				for row2 in range(0, gridSize + 1):

					#get coordinates using grid layout
					x0 = int(column1 * step)
					x1 = int(column2 * step)
					y0 = int(row1 * step)
					y1 = int(row2 * step)

					if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size

						if not (x0==y0==0 and x1==y1==resolution): #ignore full image

							#calculate height and width of bounding box
							w = x1 - x0
							h = y1 - y0

							coordsList.append([x0, y0, w, h]) #add bounding box to list

	coordsArray = np.array(coordsList)	 #format coordinates as numpy array
	num=coordsArray.shape[0]

	return coordsArray,num


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)