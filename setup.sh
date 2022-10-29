#!/bin/bash

# download artifacts
# download product-product embeddings
curl -L https://uwmadison.box.com/shared/static/sm9npu6l4w7g7gvcbdgwzo4kkqx2k05o --create-dirs --output ./data/asin2emb

# download torchscript model
curl -L  https://uwmadison.box.com/shared/static/zs2yig614n9fm8ctkwo7w117qqmg87et --create-dirs --output ./src/spinereader/artifacts/traced_donut_model_title_only.pt

# download product title embeddings
curl -L  https://uwmadison.box.com/shared/static/5pijpk17g90yqnj2ettf6q1ri7eg06bi --create-dirs --output ./src/spinereader/artifacts/title_matching_embeddings.npy

# product-product embeddings in CSV
curl -L https://uwmadison.box.com/shared/static/0bhe73xwaavvrmm43ol5lr0csjy8r7z5 --create-dirs --output ./data/books_emb_8_tidy.csv