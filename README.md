# Personal Implementations of various RL algorithms using Tensorflow 2

**Objective:**
To implement various RL algorithms using Tensorflow 2 and OpenAI Gym. More importantly, we will capture the lessons learnt as a result. Ideally, these notes really need writing up more formally in a blog format (perhaps Christmas 2020?). Note, these are my own draft notes, and thus the writing is not polished. 

As these experiments are done in my 'free' time, i.e. do not directly contribute to my PhD proper, then they're typically done in a rushed manner and code quality is not representative of my real work (I appreciate the reproducibility crisis, and rectify it in my PhD work, but not here due to time constraints).

# Deep Dive - Exploding Gradients
**Observation**:

During Task 2 (see Section "Task 2" below), the agent somewhat learns a better policy, before 'forgetting' and returning to taking seemingly random actions, then learns an even better policy and subsequently forgetting and so on. It appears that first, a sudden, relatively large dip in the loss occurs, followed by an immediate dip in performance.

**Background Theory:**

The loss function for MCPG in an episodic setting is defined as:

$J(\theta) = \mathbb{E}_{\pi_\theta}[r]$

where $J(\theta)$ is the loss function with respect to the neural network's parameters, $\theta$. $\pi_\theta$ is the agent's policy, again parameterised on $\theta$. Instead of using the typical gradient descent, we wish to use gradient _ascent_, such that we maximise the expected (discounted) future rewards.

The gradient of this loss function, or the _policy gradient_, can then be written as:

$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log{\pi_\theta(s, a) Q_{\pi_\theta}(s, a)]}$

where $s$ is the state, $a$ is the action, and $Q_{\pi_\theta}(s, a)$ is the value for taking action $a$ at state $s$, following a policy $\pi_\theta$. (taken from David Silver's RL slides https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf).

This expectation is expensive to compute, therefore we use stochastic gradient ascent to estimate the gradient instead.

Interesting note: the standard error of the mean (SEM) is $\frac{\sigma}{\sqrt{n}}$, therefore compare using a mini-batch size of 100 and 10,000, the SEM would only reduce by a factor of 10. (Goodfellow et. al 2014, p. 271). However, how does this fit in with (Smith et. al 2018)'s paper where they argue to increase the batch size instead of decaying the learning rate.

In the MCPG, a.k.a REINFORCE algorithm, the (discounted) future return $G_t$ is taken as an unbiased sample of $Q_{\pi_\theta}(s, a)$.

Typically, the gradient of the loss function w.r.t. the neural network's parameters (within a given mini-batch) is given by:

$\mathbf{\hat g} = \nabla_\theta J(\theta) =\frac{1}{m} \nabla_\theta \sum_i L(f(\mathbf{x^{(i)}}; \mathbf{\theta}), y^{(i)})$

In our MCPG algorithm, we define the loss function, $L(\cdot)$ as

$L(f(\mathbf{x^{(i)}}; \mathbf{\theta}), y^{(i)}) = -\log \pi_\theta (s, a) G_t$

Note the negative due to the use of gradient ascent instead of descent.

Thus we can write:

$\Delta \theta_t = \alpha \mathbf{\hat g}$

**Five Why's:**

Now with the background theory, we can attempt to answer Five Why's to try and identify root cause (and thus corrective action to take).

1. Why are we experiencing exploding gradients?

    - The gradient of the loss, $\mathbf{\hat g}$ is too high. My understanding is that this could be _either_ due to the gradients themselves ($\nabla_\theta$), or the magnitude of the loss $L(\cdot)$ as well.

    - In Task 2, the magnitude of the loss spikes at -25,000 at Step 8816. This causes(? - not sure of causality direction) the magnitude of the gradients to oscillate with larger amplitudes, followed by an explosion (nan values) at about Step 9200.

    - Due to the uncertainty in the first bullet point. regarding root cause, it is best to branch out the 5 Whys.

2. If exploding gradients is due to magnitude of $\nabla_\theta$ being too large, why is it too large?

    - From (Goodfellow et. al 2014, p. 281) (Deep Learning textbook), there can be "cliffs" that occur in neural networks that arise from large weights being multiplied together. When at a very steep cliff, it is akin to jumping off the cliff. The gradient ascent update then "catapults" the parameters quite far away, undoing most of the previous learning done.

    2.1 Why are there large weights being multiplied together?

    - Likely due to $\Delta \theta_t = \alpha \mathbf{\hat g}$ just gradually increasing the weights of the neural network more and more. I do not see any evidence of this in the histogram of the neural network.

    **Corrective Action**: 
    
    If Root Cause if due to 2) then we should apply some regularisation to the neural network's weights.

    1. Apply $L_1$ or $L_2$ regularisation to the weights
    2. Apply Batch Normalisation to the weights
    3. If 1) and 2) fail, then apply Gradient Clipping as a last resort.

3. If exploding gradients is due to magnitude of $L(\cdot)$ being too large, why is it too large?

    - From $L(f(\mathbf{x^{(i)}}; \mathbf{\theta}), y^{(i)}) = -\log \pi_\theta (s, a) G_t$, then the only variable that can explode is $\log \pi_\theta(s, a)$, since $G_t$ is dictated by the environment as should be well-behaved.

    - $\log \pi_\theta(s, a)$ is largest when $\pi_\theta(s, a)$ is close to 0.

    3.1. Why would $\pi_\theta(s, a)$ be close to 0???

    - This would mean that the agent is really unsure of which action to take and assigns essentially a 0 probability to each action.

        3.1.1. Why would the probability of each action be 0?

        - Could the variance neuron be predicting very large variances???


## Task 1: CartPole (Monte Carlo Policy Gradient on Discrete Action Spaces) - 17th Oct. 2020

**Summary:**

This agent tackles the `CartPole-v0` gym environment using MCPG and successfully learns to solve the problem in approx. 200 episodes. The agent completes 10,000 episodes within 5000s (clearly 10,000 is overkill).

Much love to Phil Tabor and his YouTube channel, which heavily influenced my code:
https://www.youtube.com/watch?v=mA9rxgOQyE4&ab_channel=MachineLearningwithPhil

The code for this can be found in `src/cartpole/vanilla_policy_gradient_barebones.ipynb`.

**Research Questions:**
1. How does $\gamma$ affect learning?
2. What is the "definition" of exploding gradients? What magnitude is considered "exploding", or do we simply wait til we see 'nans'.

**Lessons Learnt:**
1. The gradients are relatively well behaved (at least, compared with Task 2). However, they still get rather large, in the couple of 100's, and approching 1000.
2. Typically speaking, when the gradients spike, this is followed by a drop in performance (which may or may not be lagged).

## Task 2: InvertedPendulum-v2 (Monte Carlo Policy Gradient on Continuous Action Spaces) - 24th Oct. 2020

**Summary**:
This agent tackles the `InvertedPendulum-v2` gym environment using the basic MCPG algorithm (without additional tricks like gradient clipping etc.). The agent does reasonably well, and with luck, can briefly obtain the max. reward of 1000. However, we see intermittent drops in performance, which is hypothesised to be due to exploding gradients. 

The code for this can be found in `src/inverted_pendulum/vanilla_policy_gradient_continuous.ipynb`

**Next Steps:**

1. Explore best practices to tackle the exploding gradients problem. Ideas:

    1. Clip the gradients
    2. Different activation functions
    3. Batch normalisation
    4. Identity functions(?) similar to residual blocks etc.
    5. Parameter initialisation(is it useful in this context?)
    6. Different optimisation techniques(?)
    7. Normalise the rewards?
    8. Normalise the loss?
    9. Early stopping? "An optimisation algorithm is considered to have converged when the gradient becomes very small."
    10. Regularise the weights with L1 or L2 norm etc.
    11. Plot norm of the gradient to identify local minima/maxima. It should decrease towards 0 if a critical point.
    12. Check ratio of magnitude of parameter gradients: magnitude of parameters. Should be about 1% (see Bottou 2015)

**Research Questions:**
1. How does $\gamma$ affect learning?
2. How can we prevent exploding gradients?
3. Given that NN's assume i.i.d data, yet we loop from $t = 1$ to $t = T$ sequentially:

    a. Is it beneficial to shuffle this data _within_ an episode?
    
    b. Is it beneficial to shuffle this data _across_ episodes? (i.e. can we do Experience Replay here?)
4. If 3. is True, how can we detect or diagnose when we have non-i.i.d. data? What symptoms would be present?

**Lessons Learnt:**
1. Interestingly, I trained an agent for 20k episodes for fun, and it consistently hit the max. reward of 1000 from episodes 3000 to 5500 or so. However, after this, the agent resorted back to taking completely random actions from episode 6000 to 20000...

- I suspect this is due to a few spikes in gradients which pushed the parameters far far away from a reasonable policy, and it was never able to recover. Is the easiest way just to implement early stopping? Is there any literature on early stopping?