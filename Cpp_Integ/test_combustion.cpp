#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  std::cout << "ok\n";
  std::vector<torch::jit::IValue> inputs;

  //double data[] = {454.66, 58.669};
  torch::Tensor tensor = torch::zeros({1,2}, torch::dtype(torch::kFloat32));
  tensor[0][0] = -0.8037;
  tensor[0][1] = -0.2691;
  inputs.push_back(tensor);
  std::cout<<inputs << '\n';

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module->forward(inputs).toTensor();

  std::cout << output << '\n';
  return 0;
}
