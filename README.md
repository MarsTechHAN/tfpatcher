# tfpatcher
A ugly written tool for patching the tensorflow forzen protobuf file for compatibility to RKNN &amp; SNPE SDK.

The program will do following things:
1. Check for 'explicit_paddings' attribution and removed. SNPE's built-in tf cannot read.
2. Check for 'reduce_dim' attribution for Mean operator, if it is true (reduce demension from [1,1,1,1280] to [1, 1280]), changed it to false, and add reshape function after it. Also, all '_output_shapes' attribution will be removed, since the shapes have changed.  SNPE dont support Mean with reduce_dim=true.
3. Check for 'FusedBatchNormV3' operator, and replace it with 'FusedBatchNorm' and remove 'U' attr. Both SNPE and RKNN dont support it yet.
4. Check for 'swish_f32', and replace with a 'Sigmoid' and 'mul', both SNPE and RKNN don't read and interpret the node_def section. 

Test working networks:
- efficientnet_b0
- efficientnet_b1
- mnasnet-a1
- mobilenet_v1
- mobilenet_v2
- nasnet_a_mobile
- resnet_v2

Usage:
```
#First run will convert pb to pbtxt
python3 .\frozenpb_patcher.py .\efficientnet_b0.pb

#Second run will do the patching.
python3 .\frozenpb_patcher.py .\efficientnet_b0.pb
#python3 .\frozenpb_patcher.py .\nasnet_a_mobile.pb nasnet (nasnet has 1056 channels, unlink common 1280 channels)

```
