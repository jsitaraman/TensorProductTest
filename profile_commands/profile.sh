nsys profile \
  --stats=true \
  --trace=cuda,osrt,nvtx \
  -o tensor_product_profile \
  ./tensor_product_volume --gpu
