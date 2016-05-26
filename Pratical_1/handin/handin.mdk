Title         : Practical 1
Author        : Zheng Zhaoheng
Logo          : True

[TITLE]

# Tensor
I have found 4 expressions that can extract the second column from the given Tensor t:

    local t = torch.Tensor({{1,2,3},{4,5,6},{7,8,9}})

They are shown below:

    col = t:transpose(1,2)[2]
    col = t:split(1,2)[2]
    col = t:select(2,2)
    col = t:sub(1,3,2,2)

# Tensor and Storage
Tensor is a particular way of viewing a Storage: a Storage only represents a chunk of memory, while the Tensor interprets
this chunk of memory as having dimensions.
