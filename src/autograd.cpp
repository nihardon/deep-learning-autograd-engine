#include "autograd.h"
#include "arena.h"
#include "ops.h" 
#include <set>

// Visits the graph so that children are processed before parents
void build_topo(std::shared_ptr<Variable> v, 
                std::set<std::shared_ptr<Variable>>& visited, 
                std::vector<std::shared_ptr<Variable>>& topo) {
    
    if (visited.find(v) == visited.end()) {
        visited.insert(v);
        // Recursively visit all children
        for (auto child : v->children) {
            build_topo(child, visited, topo);
        }
        topo.push_back(v);
    }
}

void Variable::backward() {
    // Build the correct execution order
    std::vector<std::shared_ptr<Variable>> topo;
    std::set<std::shared_ptr<Variable>> visited;
    build_topo(shared_from_this(), visited, topo);
    
    // Seed the gradient (dL/dL = 1.0)
    this->grad.fill(1.0f); 

    // Run backward pass in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    // Create output tensor
    Tensor result_data(a->data.get_shape());
    
    // Perform A + B
    ops::add(a->data, b->data, result_data);
    // Build graph
    auto out = std::make_shared<Variable>(std::move(result_data));
    std::weak_ptr<Variable> weak_out = out;
    out->children = {a, b};

    // Define backward pass
    out->backward_fn = [weak_out, a, b]() {
    if (auto locked_out = weak_out.lock()) {
        ops::add(a->grad, locked_out->grad, a->grad);
        ops::add(b->grad, locked_out->grad, b->grad);
    }
};
    return out;
}


std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
    // Forward Pass
    // Calculate output shape
    int M = a->data.get_shape()[0];
    int N = b->data.get_shape()[1];
    Tensor result_data({M, N});
    
    ops::matmul(a->data, b->data, result_data);
    // Create Node
    auto out = std::make_shared<Variable>(std::move(result_data));
    out->children = {a, b};

    // Backward Pass
    // Formula:
    // dL/dA = (dL/dOut) * B^T
    // dL/dB = A^T * (dL/dOut)
    out->backward_fn = [out, a, b]() {
        // Gradient for A
        {
            // B Transposed
            Tensor B_T({b->data.get_shape()[1], b->data.get_shape()[0]});
            ops::transpose(b->data, B_T);

            // Calculate contribution: Out.grad * B.T
            Tensor grad_A_contrib({a->grad.get_shape()});
            ops::matmul(out->grad, B_T, grad_A_contrib);
            ops::add(a->grad, grad_A_contrib, a->grad);
        }

        // Gradient for B
        {
            // A Transposed
            Tensor A_T({a->data.get_shape()[1], a->data.get_shape()[0]});
            ops::transpose(a->data, A_T);

            // Calculate contribution: A.T * Out.grad
            Tensor grad_B_contrib({b->grad.get_shape()});
            ops::matmul(A_T, out->grad, grad_B_contrib);
            ops::add(b->grad, grad_B_contrib, b->grad);        
        }
    };

    return out;
}

std::shared_ptr<Variable> relu(std::shared_ptr<Variable> a) {
    Tensor result_data = a->data.clone(); 
    
    ops::relu(result_data); 

    auto out = std::make_shared<Variable>(std::move(result_data));
    out->children = {a};

    out->backward_fn = [out, a]() {
        ops::relu_backward(out->grad, a->data, a->grad);
    };

    return out;
}