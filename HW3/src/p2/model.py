import timm, math
import torch
import torch.nn as nn


class VL_model(nn.Module):
    def __init__(self, f_dim = 1024, nhead = 8, num_layers = 4, vocab_size = 18022):
        super().__init__()
        self.f_dim = f_dim # output
        self.encoder = timm.create_model('vit_large_patch14_224_clip_laion2b', pretrained=False)

        self.embedding = nn.Embedding(vocab_size, f_dim)
        self.pe = PositionalEncoding(d_model = f_dim, dropout = 0.1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=f_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)
        self.linear = nn.Linear(f_dim, vocab_size)

        # freeze encoder weight
        self.encoder.requires_grad_(False)


    def forward(self, imgs, captions, tgt_mask, tgt_key_padding_mask):
        patch_feature = self.encoder.forward_features(imgs)[:, 1:] 
    
        tgt = self.embedding(captions) * math.sqrt(self.f_dim)
        tgt = self.pe(tgt) 
        
        out = self.decoder(tgt, patch_feature, tgt_mask = tgt_mask, 
                           tgt_key_padding_mask = tgt_key_padding_mask) 
        out = self.linear(out) 
        
        return torch.swapaxes(out, 1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=54):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



if __name__ == '__main__':
    from utils.helper import get_key_padding_mask

    model = VL_model()

    max_len = 54
    batch = 10
    imgs = torch.randn((batch, 3, 224, 224)) 
    captions = torch.zeros((batch, max_len-1)).long()
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size()[-1])
    tgt_key_padding_mask = get_key_padding_mask(captions)
    
    out = model(imgs, captions, tgt_mask, tgt_key_padding_mask)
    print(out.shape) # (batch, vocab_size, max_len-1)
   
