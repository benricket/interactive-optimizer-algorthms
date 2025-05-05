import scipy
import numpy as np
import time

class Optimizer():
    
    def __init__(self):
        self.cost_history = []
        self.x_history = []

    def optimize_strict_fun(self, fun_name, x0, params):
        """
        Perform optimization on a subset of high-dimensional functions
        """
        x0 = np.array(x0)
        if fun_name == "rosenbrock":
            self.optimize(rosenbrock,x0,params)
        elif fun_name == "sphere":
            self.optimize(sphere,x0,params)

    def optimize(self, fun, x0, params):
        """
        Perform optimization on a given cost function from a given guess.
        """
        print(f"Initial guess: {x0}, value: {fun(x0)}")
        self.cost_history = []
        self.x_history = []
        
        # Save initial point
        self.x_history.append(np.array(x0).copy())
        self.cost_history.append(fun(x0))
        
        options = {}
        method = params["method"]
        options["maxiter"] = params.get("max_iters", 10000)
        tol = params.get("tol", 1e-3)
        
        if method == "gradient_descent":
            x = np.array(x0)
            val = fun(x)
            alpha = 0.001  # Learning rate
            stop_attempts = 0
            
            for i in range(options["maxiter"]):
                # Compute gradient
                grad = scipy.optimize.approx_fprime(x, fun, 1e-8)
                
                # Perform line search to get step size alpha
                search = scipy.optimize.line_search(fun, lambda x: scipy.optimize.approx_fprime(x, fun, 1e-8), x, -grad)
                alpha = search[0] if search[0] is not None else 1e-3  # fallback alpha
                
                x_new = x - alpha * grad # ADD LINE SEARCH TO GET STEP SIZE
                val_new = fun(x_new)
                
                # Save the new point
                self.x_history.append(x_new.copy())
                self.cost_history.append(val_new)
                
                # Check for convergence
                if np.abs(val_new - val) < tol:
                    stop_attempts += 1
                    if stop_attempts > 2:
                        break
                else:
                    stop_attempts = 0
                
                # Adaptive learning rate (optional)
                #if val_new > val:
                #    alpha *= 0.5  # Reduce step size if we're not improving
                #elif stop_attempts > 0:
                #    alpha *= 0.9  # Gently reduce as we approach minimum
                
                # Update guess
                x = x_new
                val = val_new
                print(f"iteration {i}")
                
        elif method == "newton":
            x = np.array(x0)
            val = fun(x)
            stop_attempts = 0
            for i in range(options["maxiter"]):
                # Compute Hessian and gradient
                # This is intentionally slow to demonstrate why approximate
                # algorithms are more useful 
                grad = scipy.optimize.approx_fprime(x, fun)
                if np.linalg.norm(grad) < tol:
                    print("Stopping due to gradient tolerance!")
                    break

                hess = scipy.differentiate.hessian(fun, x).ddf
                #print(f"Hess: {hess}")
                if np.linalg.det(hess) == 0:
                    # If Hessian is singular, add small value to diagonal
                    hess += np.eye(len(x)) * 1e-6
                    
                hess_inv = np.linalg.inv(hess)

                # Perform Newton's method
                #gamma = 0.5

                search = scipy.optimize.line_search(fun, lambda x: scipy.optimize.approx_fprime(x, fun, 1e-8), x, -hess_inv@grad)
                alpha = search[0] if search[0] is not None else 1e-2

                x_new = x - alpha * hess_inv @ grad
                val_new = fun(x_new)
                
                # Save the new point
                self.x_history.append(x_new.copy())
                self.cost_history.append(val_new)
                
                if np.abs(val_new - val) < tol:
                    stop_attempts += 1
                    if stop_attempts > 2:
                        break
                else:
                    stop_attempts = 0

                # Update guess
                x = x_new
                val = val_new

        elif method == "BFGS":
            ans = scipy.optimize.minimize(fun, x0, options=options, method="BFGS", tol=tol, callback=self.callback)
            x = ans.x
            print(f"BFGS optimization completed in {ans.nit} iterations")

        elif method == "trust-constr":
            ans = scipy.optimize.minimize(fun, x0, options=options, method="trust-constr", tol=tol, callback=self.callback)
            x = ans.x
            print(f"trust-constr optimization completed in {ans.nit} iterations")
        else:
            print(f"Method {method} not implemented.")
        
        self.x_history.append(x)

    def callback(self, x, *args):
        """
        Callback function to add guess to history
        """
        self.x_history.append(x.copy().tolist())
        print(x)
        self.cost_history.append([0])

    def callback_newton(self, x, val):
        """
        Callback function to add guess to history
        """
        self.cost_history.append(val)
        print(val)
        self.x_history.append(x.copy().tolist())

def rosenbrock(x):
    return scipy.optimize.rosen(x)

def sphere(x):
    #print(f"sphere: {np.array(np.sum(x**4))}")
    if len(x.shape) > 1:
        return np.sum(x**4,axis=1)
    return np.sum(x**4)

if __name__ == "__main__":
    opt = Optimizer()
    guess = np.array([0 for _ in range(30)])
    
    # Simple quadratic function
    def test_func(xy):
        x, y = xy
        return (x - 1)**2 + (y - 2)**2 + 3
    
    t0 = time.time()
    opt.optimize(rosenbrock, guess, params={"method": "trust-constr"})
    print(f"Time: {time.time() - t0}")
    print(f"Final point: {opt.x_history[-1]}")
    print(f"Final value: {rosenbrock(opt.x_history[-1])}")
    


