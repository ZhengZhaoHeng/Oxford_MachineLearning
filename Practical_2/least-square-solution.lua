require 'torch'

data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

X = data:clone()
X:select(2, 1):fill(1)
y = data:select(2, 1)
theta = torch.inverse((X:t() * X)) * X:t() * y

print(theta)
dataTest = torch.Tensor{
   {6, 4},
   {10, 5},
   {14, 8}
}

print('id  approx')
for i = 1,(#dataTest)[1] do
   local myPrediction = theta[{{2,3}}] * dataTest[i][{{1,2}}] + theta[1]
   print(string.format("%2d  %6.2f", i, myPrediction))
end
