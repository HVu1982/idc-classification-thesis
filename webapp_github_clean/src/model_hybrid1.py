import math
import torch
import torch.nn as nn
import torch.nn.init as init  # Để dùng init functions
import torchvision.models as models
from timm.models.vision_transformer import Block, trunc_normal_

class CNNDeiTSmall(nn.Module):
    def __init__(self, num_classes=2, resnet_variant="resnet50", img_size=(224, 224),
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.,  # ⬆️ Tăng capacity
                 qkv_bias=True, drop_rate=0.2, attn_drop_rate=0.1, drop_path_rate=0.2,  # ⬆️ Tăng regularization
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = img_size

        # 1. CNN Backbone (ResNet) với Frozen BatchNorm
        if resnet_variant == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            cnn_out_dim = 512
        elif resnet_variant == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            cnn_out_dim = 512
        elif resnet_variant == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            cnn_out_dim = 2048
        else:
            raise ValueError("Unsupported ResNet variant")

        # bỏ avgpool + fc
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 🆕 Freeze BatchNorm layers để ổn định training
        self._freeze_bn()

        # 2. 🆕 Adaptive Projection với residual connection
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        # 🆕 Skip connection nếu cần giảm dimension
        self.skip_proj = nn.Linear(cnn_out_dim, embed_dim) if cnn_out_dim != embed_dim else nn.Identity()

        # 3. Class token + Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14*14 + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 4. Transformer Encoder (Stochastic Depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 5. 🆕 Enhanced Classifier Head với bottleneck
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate * 1.5),  # Dropout cao hơn ở head
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Init weights
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _freeze_bn(self):
        """Freeze BatchNorm layers trong ResNet để ổn định training"""
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.BatchNorm2d):
                #module.eval()        # ✅ Chỉ freeze weights, KHÔNG gọi .eval()
                for param in module.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, H, W):
        """Nội suy pos_embed để khớp số patch (H*W)"""
        num_patches = H * W
        N = self.pos_embed.shape[1] - 1

        if num_patches == N:
            return self.pos_embed

        cls_pos = self.pos_embed[:, 0:1, :]
        pos_tokens = self.pos_embed[:, 1:, :]
        dim = pos_tokens.shape[-1]

        gs_old = int(math.sqrt(N))
        pos_tokens = pos_tokens.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)

        pos_tokens = nn.functional.interpolate(
            pos_tokens, size=(H, W), mode='bicubic', align_corners=False
        )

        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, num_patches, dim)
        new_pos_embed = torch.cat((cls_pos, pos_tokens), dim=1)
        return new_pos_embed

    def forward_features(self, x):
        B = x.shape[0]
        
        # CNN feature extraction
        feats = self.feature_extractor(x)  # (B, C, H_feat, W_feat)
        H, W = feats.shape[2], feats.shape[3]

        # Flatten and project
        feats_flat = feats.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 🆕 Residual projection
        feats_proj = self.proj(feats_flat)  # (B, H*W, embed_dim)
        feats_skip = self.skip_proj(feats_flat)
        feats = feats_proj + feats_skip  # Residual connection

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feats = torch.cat((cls_tokens, feats), dim=1)

        # Add positional encoding
        pos_embed = self.interpolate_pos_encoding(feats, H, W)
        feats = feats + pos_embed
        feats = self.pos_drop(feats)

        # Transformer blocks
        for blk in self.blocks:
            feats = blk(feats)
        feats = self.norm(feats)
        
        return feats[:, 0]  # Return cls token

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

    def train(self, mode=True):
        """Override train() để giữ BatchNorm frozen"""
        super().train(mode)
        if mode:
            self._freeze_bn()
        return self


#if __name__ == "__main__":
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    model = CNNDeiTSmall(
#        num_classes=2, 
#        resnet_variant="resnet50",
#        embed_dim=512,  # Tăng từ 384
#        depth=8,        # Tăng từ 6
#        num_heads=8,    # Tăng từ 6
#        drop_rate=0.2,
#        drop_path_rate=0.2
#    ).to(device)
    
#    dummy_input = torch.randn(4, 3, 224, 224).to(device)
#    output = model(dummy_input)
#    print("✅ Output shape:", output.shape)
#    print(f"📊 Total params: {sum(p.numel() for p in model.parameters()):,}")