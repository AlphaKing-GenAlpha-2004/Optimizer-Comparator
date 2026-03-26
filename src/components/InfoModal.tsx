import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Info, X, Zap, Activity, TrendingDown, Target, Brain, Layers, RefreshCw, BarChart3 } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const InfoModal: React.FC<InfoModalProps> = ({ isOpen, onClose }) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[500] flex items-center justify-center p-4 bg-black/60 backdrop-blur-md">
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="bg-white rounded-[2.5rem] p-8 md:p-12 max-w-4xl w-full shadow-2xl border border-[#E7E5E4] max-h-[90vh] flex flex-col overflow-hidden relative"
          >
            {/* Close Button */}
            <button
              onClick={onClose}
              className="absolute top-8 right-8 p-3 hover:bg-[#F5F5F4] rounded-full transition-all duration-300 group z-10"
              aria-label="Close modal"
            >
              <X className="w-6 h-6 text-[#78716C] group-hover:rotate-90 transition-transform duration-300" />
            </button>

            {/* Header */}
            <div className="flex items-center gap-4 mb-10">
              <div className="w-14 h-14 rounded-2xl bg-emerald-50 flex items-center justify-center">
                <Brain className="w-8 h-8 text-emerald-600" />
              </div>
              <div>
                <h2 className="text-3xl font-bold tracking-tight text-[#1C1917]">Optimizer Intelligence Guide</h2>
                <p className="text-[#78716C] font-medium">Mastering the mechanics of neural network training</p>
              </div>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto pr-6 space-y-16 custom-scrollbar pb-10">
              
              {/* SECTION 1: INTRODUCTION TO OPTIMIZERS */}
              <section>
                <div className="flex items-center gap-3 mb-6">
                  <Layers className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">01. Introduction to Optimizers</h3>
                </div>
                <div className="space-y-4 text-[#44403C] leading-relaxed">
                  <p>
                    A <strong>Neural Network</strong> is a computational model inspired by the human brain, consisting of layers of interconnected "neurons." Each connection has a <strong>Weight</strong> (strength) and each neuron has a <strong>Bias</strong> (offset).
                  </p>
                  <p>
                    When we start training, these weights and biases are <strong>initialized randomly</strong>. Because they are random, the network's initial predictions are almost always wrong. To measure "how wrong" the network is, we use a <strong>Loss Function</strong> (also called a Cost Function).
                  </p>
                  <p>
                    The goal of training is to find the specific set of weights and biases that result in the lowest possible loss. This is where the optimizer comes in.
                  </p>
                  <div className="bg-emerald-50 border-l-4 border-emerald-500 p-6 rounded-r-2xl my-6">
                    <p className="text-emerald-900 font-semibold italic">
                      "An optimizer is an algorithm that updates the weights and biases of a neural network in order to minimize the loss function."
                    </p>
                  </div>
                  <p>
                    The training loop follows a cycle: The network makes a prediction (Forward Pass), calculates the error (Loss), determines the direction to reduce that error (Backpropagation), and finally, the <strong>Optimizer</strong> adjusts the weights based on those findings.
                  </p>
                </div>
              </section>

              {/* SECTION 2: ROLE OF OPTIMIZERS */}
              <section>
                <div className="flex items-center gap-3 mb-6">
                  <Target className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">02. Role of Optimizers</h3>
                </div>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <p className="text-[#44403C]">Optimizers are the "drivers" of the learning process. They don't just change weights; they control <em>how</em> those changes happen, affecting:</p>
                    <ul className="space-y-3">
                      <li className="flex gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0" />
                        <span><strong>Speed of Convergence:</strong> How many epochs it takes to reach a good solution.</span>
                      </li>
                      <li className="flex gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0" />
                        <span><strong>Stability:</strong> Whether the loss decreases smoothly or fluctuates wildly.</span>
                      </li>
                      <li className="flex gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0" />
                        <span><strong>Accuracy:</strong> The final performance level the model can achieve.</span>
                      </li>
                      <li className="flex gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0" />
                        <span><strong>Generalization:</strong> How well the model performs on new, unseen data.</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-[#F5F5F4] p-6 rounded-3xl border border-[#E7E5E4]">
                    <h4 className="font-bold mb-3 text-sm uppercase tracking-widest text-[#78716C]">The Trade-off</h4>
                    <p className="text-sm text-[#44403C] leading-relaxed">
                      Choosing the wrong optimizer can lead to poor results, even with a perfect model architecture. Some optimizers are fast but unstable; others are slow but reliable. Finding the right balance is a core skill in Deep Learning.
                    </p>
                  </div>
                </div>
              </section>

              {/* SECTION 3: CORE TERMINOLOGY */}
              <section>
                <div className="flex items-center gap-3 mb-8">
                  <Activity className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">03. Core Terminology</h3>
                </div>
                <div className="space-y-8">
                  {[
                    {
                      term: "Gradient",
                      simple: "The direction and steepness of the hill we are walking down.",
                      tech: "The vector of partial derivatives of the loss function with respect to the weights. It points in the direction of steepest ascent."
                    },
                    {
                      term: "Learning Rate (η)",
                      simple: "The size of the step we take in the direction of the gradient.",
                      tech: "A hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function."
                    },
                    {
                      term: "Loss Function",
                      simple: "A score that tells us how far off our predictions are from the truth.",
                      tech: "A mathematical function that maps values of one or more variables onto a real number intuitively representing some 'cost' associated with the event."
                    },
                    {
                      term: "Epoch",
                      simple: "One complete pass through the entire training dataset.",
                      tech: "A single iteration through the entire training set during the training of a machine learning model."
                    },
                    {
                      term: "Batch / Mini-batch",
                      simple: "A small chunk of data used to update the weights instead of the whole set at once.",
                      tech: "A subset of the training data used to compute the gradient and update the model parameters in one iteration."
                    },
                    {
                      term: "Weights & Biases",
                      simple: "The internal 'knobs' the model turns to learn patterns.",
                      tech: "The learnable parameters of a neural network. Weights determine the influence of input features; biases provide an offset to the activation function."
                    },
                    {
                      term: "Backpropagation",
                      simple: "The process of working backward from the error to find out which weights caused it.",
                      tech: "An algorithm for calculating the gradient of the loss function with respect to the weights by applying the chain rule of calculus."
                    },
                    {
                      term: "Convergence",
                      simple: "When the model stops learning significantly because it has found a good solution.",
                      tech: "The state where the loss function has reached a local or global minimum and further training does not improve performance."
                    },
                    {
                      term: "Overfitting",
                      simple: "When the model memorizes the training data perfectly but fails on new data.",
                      tech: "When a model learns the noise and details in the training data to the extent that it negatively impacts the performance of the model on new data."
                    },
                    {
                      term: "Underfitting",
                      simple: "When the model is too simple to learn the patterns in the data.",
                      tech: "When a model cannot capture the underlying trend of the data, resulting in poor performance on both training and test sets."
                    },
                    {
                      term: "Momentum",
                      simple: "Using the 'speed' from previous steps to help push through flat areas or small bumps.",
                      tech: "A technique that adds a fraction of the previous weight update to the current update to accelerate gradient descent in the relevant direction."
                    },
                    {
                      term: "Adaptive Learning Rate",
                      simple: "Automatically changing the step size for each individual weight based on its history.",
                      tech: "Algorithms that adjust the learning rate for each parameter individually, typically based on the magnitude of past gradients."
                    },
                    {
                      term: "Gradient Descent",
                      simple: "The standard way of finding the bottom of the loss 'valley'.",
                      tech: "A first-order iterative optimization algorithm for finding a local minimum of a differentiable function."
                    }
                  ].map((item, i) => (
                    <div key={i} className="group border-b border-[#E7E5E4] pb-6 last:border-0">
                      <h4 className="text-lg font-bold text-[#1C1917] mb-2 group-hover:text-emerald-600 transition-colors">{item.term}</h4>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-600 block mb-1">Intuition</span>
                          <p className="text-sm text-[#44403C]">{item.simple}</p>
                        </div>
                        <div>
                          <span className="text-[10px] font-bold uppercase tracking-widest text-[#78716C] block mb-1">Technical</span>
                          <p className="text-sm text-[#78716C] italic">{item.tech}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* SECTION 4: GENERAL TRAINING PIPELINE */}
              <section>
                <div className="flex items-center gap-3 mb-8">
                  <RefreshCw className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">04. General Training Pipeline</h3>
                </div>
                <div className="relative">
                  <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-[#E7E5E4]" />
                  <div className="space-y-10">
                    {[
                      "Initialize weights and biases randomly.",
                      "Perform a Forward Pass: Input data through the network to get predictions.",
                      "Compute Loss: Compare predictions with actual labels using a loss function.",
                      "Backpropagation: Calculate the gradient of the loss for every parameter.",
                      "Optimizer Update: Adjust weights and biases using the calculated gradients.",
                      "Repeat for all batches in the dataset.",
                      "Repeat for multiple Epochs until convergence."
                    ].map((step, i) => (
                      <div key={i} className="relative pl-12">
                        <div className="absolute left-0 top-1 w-8 h-8 rounded-full bg-white border-2 border-emerald-500 flex items-center justify-center z-10">
                          <span className="text-xs font-bold text-emerald-600">{i + 1}</span>
                        </div>
                        <p className="text-[#44403C] font-medium">{step}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              {/* SECTION 5: OPTIMIZERS (VERY DETAILED) */}
              <section>
                <div className="flex items-center gap-3 mb-10">
                  <Zap className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">05. Deep Dive into Optimizers</h3>
                </div>
                
                <div className="space-y-20">
                  {/* SGD */}
                  <div className="space-y-6">
                    <div className="flex items-baseline gap-4">
                      <h4 className="text-2xl font-bold text-[#1C1917]">SGD <span className="text-sm font-normal text-[#78716C] ml-2">(Stochastic Gradient Descent)</span></h4>
                    </div>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <p className="text-sm text-[#44403C]"><strong>Concept:</strong> The most fundamental optimizer. It updates parameters in the opposite direction of the gradient.</p>
                        <p className="text-sm text-[#44403C]"><strong>Intuition:</strong> Walking straight down a hill. If the hill is steep, you take a big step; if it's flat, you take a small step.</p>
                        <div className="bg-[#1C1917] text-emerald-400 p-4 rounded-xl font-mono text-xs">
                          θ = θ − η · ∇J(θ)
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-emerald-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-emerald-700 uppercase">Pros</span>
                            <p className="text-xs text-emerald-900">Simple, low memory, can generalize well.</p>
                          </div>
                          <div className="bg-red-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-red-700 uppercase">Cons</span>
                            <p className="text-xs text-red-900">Slow, can get stuck in local minima, oscillates.</p>
                          </div>
                        </div>
                        <p className="text-xs text-[#78716C]"><strong>When to use:</strong> Simple models or when fine-tuning a pre-trained model for better generalization.</p>
                      </div>
                    </div>
                  </div>

                  {/* Adagrad */}
                  <div className="space-y-6">
                    <div className="flex items-baseline gap-4">
                      <h4 className="text-2xl font-bold text-[#1C1917]">Adagrad <span className="text-sm font-normal text-[#78716C] ml-2">(Adaptive Gradient Algorithm)</span></h4>
                    </div>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <p className="text-sm text-[#44403C]"><strong>Concept:</strong> It adapts the learning rate to the parameters, performing larger updates for infrequent parameters and smaller updates for frequent ones.</p>
                        <p className="text-sm text-[#44403C]"><strong>Intuition:</strong> Slowing down on paths you've walked many times before to ensure you don't overshoot the bottom.</p>
                        <div className="bg-[#1C1917] text-emerald-400 p-4 rounded-xl font-mono text-xs">
                          G = G + g²<br/>
                          θ = θ − (η / √(G + ε)) · g
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-emerald-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-emerald-700 uppercase">Pros</span>
                            <p className="text-xs text-emerald-900">No manual LR tuning needed for each parameter.</p>
                          </div>
                          <div className="bg-red-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-red-700 uppercase">Cons</span>
                            <p className="text-xs text-red-900">LR eventually becomes too small, stopping learning.</p>
                          </div>
                        </div>
                        <p className="text-xs text-[#78716C]"><strong>When to use:</strong> Sparse data (like NLP or recommendation systems).</p>
                      </div>
                    </div>
                  </div>

                  {/* RMSProp */}
                  <div className="space-y-6">
                    <div className="flex items-baseline gap-4">
                      <h4 className="text-2xl font-bold text-[#1C1917]">RMSProp <span className="text-sm font-normal text-[#78716C] ml-2">(Root Mean Square Propagation)</span></h4>
                    </div>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <p className="text-sm text-[#44403C]"><strong>Concept:</strong> An unpublished optimizer (proposed by Geoff Hinton) that fixes Adagrad's disappearing learning rate by using a moving average of squared gradients.</p>
                        <p className="text-sm text-[#44403C]"><strong>Intuition:</strong> Smoothing out the steps so that the learning rate doesn't drop to zero prematurely.</p>
                        <div className="bg-[#1C1917] text-emerald-400 p-4 rounded-xl font-mono text-xs">
                          E[g²] = βE[g²] + (1−β)g²<br/>
                          θ = θ − (η / √(E[g²] + ε)) · g
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-emerald-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-emerald-700 uppercase">Pros</span>
                            <p className="text-xs text-emerald-900">Very stable, works well for Recurrent Neural Networks.</p>
                          </div>
                          <div className="bg-red-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-red-700 uppercase">Cons</span>
                            <p className="text-xs text-red-900">Still requires manual tuning of global LR.</p>
                          </div>
                        </div>
                        <p className="text-xs text-[#78716C]"><strong>When to use:</strong> RNNs and non-stationary objectives.</p>
                      </div>
                    </div>
                  </div>

                  {/* Adam */}
                  <div className="space-y-6">
                    <div className="flex items-baseline gap-4">
                      <h4 className="text-2xl font-bold text-[#1C1917]">Adam <span className="text-sm font-normal text-[#78716C] ml-2">(Adaptive Moment Estimation)</span></h4>
                    </div>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <p className="text-sm text-[#44403C]"><strong>Concept:</strong> The 'Gold Standard'. It combines the benefits of Momentum (moving average of gradients) and RMSProp (moving average of squared gradients).</p>
                        <p className="text-sm text-[#44403C]"><strong>Intuition:</strong> A smart navigator that remembers the general direction (momentum) and adjusts its step size based on how bumpy the terrain is (adaptive LR).</p>
                        <div className="bg-[#1C1917] text-emerald-400 p-4 rounded-xl font-mono text-xs">
                          m = β₁m + (1−β₁)g<br/>
                          v = β₂v + (1−β₂)g²<br/>
                          θ = θ − (η / (√v̂ + ε)) · m̂
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-emerald-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-emerald-700 uppercase">Pros</span>
                            <p className="text-xs text-emerald-900">Fast, robust, works for almost everything.</p>
                          </div>
                          <div className="bg-red-50 p-3 rounded-xl">
                            <span className="text-[10px] font-bold text-red-700 uppercase">Cons</span>
                            <p className="text-xs text-red-900">Computationally expensive, might overfit.</p>
                          </div>
                        </div>
                        <p className="text-xs text-[#78716C]"><strong>When to use:</strong> Default choice for most Deep Learning tasks.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              {/* SECTION 6: INTUITIVE ANALOGIES */}
              <section>
                <div className="flex items-center gap-3 mb-8">
                  <Brain className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">06. Intuitive Analogies</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {[
                    { name: "SGD", desc: "Walking downhill step-by-step, blindly following the local slope.", color: "bg-blue-50 text-blue-700" },
                    { name: "Adagrad", desc: "Slowing down on familiar paths to avoid missing the subtle details.", color: "bg-amber-50 text-amber-700" },
                    { name: "RMSProp", desc: "Smoothing out the steps to maintain a steady pace on uneven ground.", color: "bg-purple-50 text-purple-700" },
                    { name: "Adam", desc: "Smart navigation using memory of the path and awareness of the terrain.", color: "bg-emerald-50 text-emerald-700" }
                  ].map((item, i) => (
                    <div key={i} className={cn("p-6 rounded-3xl border border-black/5", item.color)}>
                      <h4 className="font-bold mb-2">{item.name}</h4>
                      <p className="text-xs leading-relaxed opacity-80">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* SECTION 7: COMPARISON SUMMARY */}
              <section>
                <div className="flex items-center gap-3 mb-8">
                  <BarChart3 className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">07. Comparison Summary</h3>
                </div>
                <div className="overflow-hidden rounded-3xl border border-[#E7E5E4]">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-[#F5F5F4]">
                        <th className="p-4 text-xs font-bold uppercase tracking-widest text-[#78716C]">Optimizer</th>
                        <th className="p-4 text-xs font-bold uppercase tracking-widest text-[#78716C]">Speed</th>
                        <th className="p-4 text-xs font-bold uppercase tracking-widest text-[#78716C]">Stability</th>
                        <th className="p-4 text-xs font-bold uppercase tracking-widest text-[#78716C]">Best Use</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#E7E5E4]">
                      {[
                        { name: "SGD", speed: "Slow", stability: "Low", use: "Fine-tuning" },
                        { name: "Adagrad", speed: "Moderate", stability: "High", use: "Sparse Data" },
                        { name: "RMSProp", speed: "Fast", stability: "High", use: "RNNs" },
                        { name: "Adam", speed: "Very Fast", stability: "High", use: "General Purpose" }
                      ].map((row, i) => (
                        <tr key={i} className="hover:bg-[#FAFAF9] transition-colors">
                          <td className="p-4 text-sm font-bold">{row.name}</td>
                          <td className="p-4 text-sm text-[#44403C]">{row.speed}</td>
                          <td className="p-4 text-sm text-[#44403C]">{row.stability}</td>
                          <td className="p-4 text-sm text-[#44403C]">{row.use}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* SECTION 8: PRACTICAL INSIGHTS */}
              <section>
                <div className="flex items-center gap-3 mb-8">
                  <Target className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">08. Practical Insights</h3>
                </div>
                <div className="space-y-6">
                  <div className="p-6 bg-white border border-[#E7E5E4] rounded-3xl shadow-sm">
                    <h4 className="font-bold text-[#1C1917] mb-2">Why Adam often performs best?</h4>
                    <p className="text-sm text-[#44403C] leading-relaxed">Adam combines the best of both worlds: it handles noisy gradients well (like Momentum) and deals with different scales of gradients (like RMSProp). This makes it highly versatile across many different types of data and architectures.</p>
                  </div>
                  <div className="p-6 bg-white border border-[#E7E5E4] rounded-3xl shadow-sm">
                    <h4 className="font-bold text-[#1C1917] mb-2">Why SGD sometimes generalizes better?</h4>
                    <p className="text-sm text-[#44403C] leading-relaxed">Because SGD is "noisier" and slower, it often finds broader, flatter minima in the loss landscape. These flatter minima are known to generalize better to new data compared to the sharp minima sometimes found by faster adaptive optimizers.</p>
                  </div>
                </div>
              </section>

              {/* SECTION 9: CONNECTION TO YOUR APP */}
              <section>
                <div className="flex items-center gap-3 mb-8">
                  <TrendingDown className="w-6 h-6 text-emerald-600" />
                  <h3 className="text-xl font-bold text-[#1C1917] uppercase tracking-wider">09. Connection to Your App</h3>
                </div>
                <div className="space-y-6 text-[#44403C] leading-relaxed">
                  <p>In <strong>Neur-O-Opt Lab</strong>, you can observe these theories in real-time through the following visualizations:</p>
                  <ul className="grid md:grid-cols-2 gap-6">
                    <li className="p-4 bg-[#F5F5F4] rounded-2xl">
                      <span className="font-bold block mb-1">Loss & Accuracy Graphs</span>
                      <span className="text-xs">Watch how quickly each optimizer drives the error down and the precision up.</span>
                    </li>
                    <li className="p-4 bg-[#F5F5F4] rounded-2xl">
                      <span className="font-bold block mb-1">Gradient Norm</span>
                      <span className="text-xs">See the magnitude of the 'hill steepness' over time. High norms indicate aggressive learning.</span>
                    </li>
                    <li className="p-4 bg-[#F5F5F4] rounded-2xl">
                      <span className="font-bold block mb-1">Update Ratio</span>
                      <span className="text-xs">Observe the ratio of weight changes. This helps identify if an optimizer is becoming too slow (Adagrad) or unstable.</span>
                    </li>
                    <li className="p-4 bg-[#F5F5F4] rounded-2xl">
                      <span className="font-bold block mb-1">Convergence Rate</span>
                      <span className="text-xs">Compare the efficiency of adaptive methods against the baseline SGD.</span>
                    </li>
                  </ul>
                </div>
              </section>

            </div>

            {/* Footer */}
            <div className="mt-8 pt-8 border-t border-[#E7E5E4] flex justify-between items-center">
              <p className="text-xs text-[#78716C] font-medium uppercase tracking-widest">Neur-O-Opt Lab • Educational Module</p>
              <button
                onClick={onClose}
                className="px-8 py-3 bg-[#1C1917] text-white rounded-2xl font-bold hover:bg-emerald-600 transition-all duration-300 shadow-lg shadow-black/10"
              >
                Got it, let's train!
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default InfoModal;
