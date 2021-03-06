require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  
  -- ...something here...
  self.output = torch.add(input, torch.abs(input)):div(2):pow(2):clone()
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- ...something here...
  local sign = torch.sign(input)
  self.gradInput = sign:add(torch.abs(sign)):cmul(input):cmul(gradOutput):clone()
  return self.gradInput
end

