require 'torch'
require 'math'
require 'nngraph'

local function create_model()

	local x1 = nn.Identity()()
	local x2 = nn.Identity()()
	local x3 = nn.Identity()()

	-- x1, x2: 10 * 1
	-- x3: 20 * 1

	local x3_linear = nn.Linear(20, 10)(x3)
	local cmul = nn.CMulTable()({x2, x3_linear})
	local output = nn.CAddTable()({x1, cmul})

	local model = nn.gModule({x1, x2, x3}, {output})
	return model
end

model = create_model()
--graph.dot(model.fg, 'Big MLP')
x1 = torch.rand(10)
x2 = torch.rand(10)
x3 = torch.rand(20)
print(x1)
print(x2)
print(x3)

output = model:forward({x1, x2, x3})

print(output)