from sentence_transformers import SentenceTransformer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

class SentenceEncoder(ModuleAttrMixin):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        # self.normalizer = LinearNormalizer()

    def forward(self, x):
        x = self.model.encode(x)
        # x = self.normalizer(x)
        return x

    def output_shape(self):
        return self.model.encode(['']).shape
        