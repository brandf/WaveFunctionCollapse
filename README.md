
# nanoGPT
This exploration is based on the awesome [nanoGPT](https://github.com/karpathy/nanoGPT) repository. For the original README, see upstream.

# Wave Function Collapse
Wave Function Collapse (WFC) is an idea I have for how to build language models using Transformers.  It takes inspiration from a loose analogy to Quantum Mechanics, however it's not required that you understand QM to understand how WFC works. This repository both describes the idea and will attempt to implement/validate small scale models.

The goal of WFC is to be used for generative AI, similar to popular transformer based models such as ChatGPT. The concept isn't specific to text LLMs, however text is a familiar and a relatively low cost way to validate it.  

The WFC approach is a bit different from your typical decoder-only LLM in terms of architecture, training, inference, and loss, but it's compatible with the same training sets and makes use of standard Transformer blocks.  

## Understanding Decoder-only LLMs
Before we get into the details of this idea, let's do a quick review of a typical decoder-only LLM model.  If you're already an expert you can skip this section, but some of the diagrams relate to how WFC works, so it may be worth a skim anyway.

![decoder LLM diagram](assets/decoderLLM_diagram.png)

I'll assume you're already familiar with LLMs, and just point out a few relevant observations:

- The job of the Transformer blocks is to transform each embedding into the next embedding
  + at training this trains on all content lengths in parallel
  + at inference only the right most disembedding head is used to get the next token
  + embeddings are transformed from present to next in one go (through several serial blocks)
- The context window (typically much longer than shown), represents the most recent 'past' tokens only
  + there may be additional past tokens that are too far back to fit in the context window
  + as you append/shift you can think of the window as sliding along the timeline to the right
- Autoregression is happening in 'token space'
  + even though the output of the disembedding may be a distribution over our vocab, these nuances are discarded.
  + while this looks horribly inefficient, if you add a KV cache, then most of the work done by the transformer blocks is reused across autoregression steps.

If we zoom out from the details a bit we see that there are 4 'encoding spaces' that represent our input data, as well as several transformations between them:
![decoder LLM spaces](assets/decoderLLM_spaces.png)

During inference, we loop through these states in a clockwise fashion
- Starting with text
- This gets tokenized via something like byte pair encoding (for example)
- We one-hot encode the tokens to move into probability space
- We then embed these probabilities to get a sequence of points in embedding space
- The transformer blocks do prediction in embedding space
- Then we convert the next embedding to a probability distribution via the disembedding network
- Then we sample this distribution to get to a token
- And finally we decode this token back into text and output to the user

It's important to recognize a few things:

1) The blue transformations are learned, while the other red transformations are simple procedures
2) Each autoregression step, we do a full 'lap' around the various spaces
3) The transformations between spaces are sometimes lossy
4) Each space is convenient for a specific purpose
    * Text Space: useful for human input/output
      - "Hello, my name is Brandon"
    * Token Space: discrete sequence (integers) good for training storage and as a first step towards capturing patterns in the language (common text sequences become single tokens)
      - [0, 212, 43, 9192, 156, 21, 1882, 500]
    * Probability Space: useful for loss functions.  More expressive than tokens because they can represent both discrete tokens in our vocabulary (one hot encoded) as well as distributions of multiple tokens that can be sampled to pick a discrete token.  However each token is represented by many dimensions (vocabulary size), so this is more of an intermediate representation.
      - ![probability space](assets/probability_space.png)
    * Embedding Space: more compressed and semantically meaningful than Probability Space.  This is where all of the work of the Transformers is done.
      - ![embedding space](assets/embedding_space.png)

To really understand what a transformer is doing we need to understand how embedding works because that's where all of the learnable transformations take place (blue labels in the space diagram above).

For a toy example, let's use a two dimensional embedding space with a vocabulary size of 3.  This is convenient because we can visualize all three spaces in one RGB image. 

![embedding vocabulary](assets/embedding_vocab.png)

* The X/Y coordinates of a pixel in the image represent the embeddings 
  - note that we can interpolate between two points to find a path between tokens
  - a good embedding space would semantically cluster similar tokens
* Each of the color primaries are associated with a token
  - note the lossiness in going from embedding -> probability -> token
  - tokens maps to a single points (A, B, or C) in embedding space
  - going in the other direction, whole regions/subspaces can map to the same token (effectively one-hot distributions)
  - ![min entropy](assets/one_hot.png)
* RGB pixel colors illustrate probability space
  - for the certain tokens/one-hot distributions these are pure red, green, blue pixels
  - for the uncertain embeddings they are a superposition of the color primaries (red, green, blue)

Now let's consider what happens when we train different parts of the network.  The two trainable sub-networks are the embedding/disembedding transform and the transformer stack, these are typically trained together (end to end), however for illustrative purposes let's look at the effect of training them independently.

If we just train the embedding/disembedding transform:

![trained embedding](assets/embedding_vocab2.png)

* The shape of the regions has changed a bit like a lava lamp
  - at the start of training, it's like you shook up the lava lamp (random initialization)
  - over the course of training it congeals into more continuous regions and clusters in a semantically meaningful way.
* Note that we still have 3 points representing our token
  - however they map to different points in embedding space
* Embedding space would actually be more 'smooth' than this
  - I'm only showing a few levels of superposition for illustrative purposes

Now let's consider a couple points in embedding space that don't correspond to the tokens:

![embedding superposition](assets/embedding_superposition.png)

* Points D and E are in between the primary color regions
  - They are a superposition of multiple primary colors (aka tokens)
  - Point E is mostly Red, some Blue, and a little Green
  - Point D is gray'ish, about even Red, Blue, and Green - this represents a very uncertain distribution
  - ![max entropy](assets/max_entropy.png)
* We could palettized this RGB image down to just the primary colors
  - one way to do that would be to sample a primary color by treating the RGB as a probability distribution
  - this would have the effect of 'quantizing' D and E to one of the token points (A, B, or C)
  - that is exactly what happens at each autoregressive step in the decoder-only LLM (a lossy transformation)


Here is a visualization of the effect of stochastically sampling (transformations from embedding -> token).  You can see if you squint your eyes, it approximates the gradients above, but only using the 3 primary colors (tokens):

![embedding palette](assets/embedding_palette.png)

Now let's freeze the embedding space, and look at the Transformer stack, which performs next embedding prediction.
![embedding context](assets/embedding_context.png)

* You can think of the context window as a path connecting points corresponding to tokens
  - In this example, the context window contains ABC

The goal of the transformer stack is to transform from each point on the path to the next point on the path, and extrapolate one further point in the future.

![decoderLLM next token path](assets/decoderLLM_next_token_path.png)

* The black arrows show how it's learned to transform between points, and the gray arrow shows the extrapolated next embedding prediction
  - Note that they don't end up exactly at the ideal token points
  - the black arrows are used for parallel training and only the gray arrow is used at inference.
  - you can sort of think of the layers/Blocks in the Transformer stack as incrementally moving along the path, but in reality it's more like the first Block gets you most of the way and the other blocks refine it.
* The yellow arrows show the effect of the sampling
  - Note that this effectively 'quantizes' the ends of the black/gray arrows to the token points
* Sometimes there is inherent ambiguity in our training set and that is reflected in where the arrows end
  - For the gray arrow, it's saying 'typically the next word would be B, but sometimes its G or B
  - 

As the network learns end-to-end the 'lava' and the 'paths' are trying to settle into the configuration that leads to the best accuracy on our training set.

## How might we improve upon this?
First of all, let's set some stakes in the ground.  I don't want to change the 4 spaces above, or make changes to the way Transformers work.  These are too established/proven and I would like to leverage future research/industry improvements.  

Instead, this idea focuses on making relatively minor changes to the operations between these spaces, guided by some insights gleaned from a loose analogy to Quantum Mechanics.  The goal of these changes is to end up with a system that is similar to above, only better in terms of accuracy, compute, memory, scalability, and/or reasoning & planning.

Before I get into the details of the analogy let's look at some things in the model above that we could possibly improve upon:

1. The Transformer Blocks are trying to transform from an embedding to the next embedding using many sequential Blocks with independent weights (perhaps refinement between layers).
    - rather than via an iterative refinement process in embedding space (like an ODE).
    - this means it's the same cost no matter how complex the problem is.  
    - "Birds of a feather flock..." takes just as much compute as "When I opened the old book, I discovered..."
    - if you wanted to early-out of the block stack you would have to estimate when you are 'close enough'
1. The autoregression is done in token space, requiring multiple lossy transformations
    - if we were able to auto-regress in embedding space, we may be able to preserve more nuance
    - consider that multiple points in embedding space can map to the same probability distribution and that different probability distributions can be sampled resulting in the same token.
    - the conversion to/from tokens is akin to quantizing the whole embedding 'volume' down to just the N points within the volume that correspond to tokens in the vocabulary. 
1. It's just spitting out the first thing that comes to mind, with no 'thinking ahead'.
    - Speculating multiple future tokens is expensive in this model because of the butterfly effect.
    - This limits our opportunity for reasoning (steering towards preferred futures)
    - If we could speculate even with low confidence, it could potentially improve accuracy.
    - ![many worlds](assets/many_worlds.png)

## Understanding Wave Function Collapse
Now let's take a look at the diagram for Wave Function Collapse, and then I'll discuss how it incorporates these opportunities, and I'll explain some of the insights that led to it.

![wfc diagram](assets/wavefunctioncollapse_diagram.png)

Overall this should look familiar - the goal is to make minor changes after all.  It starts out identical to a standard decoder LLM, however the Transformer is a bit different.
1. The input embeddings are divided into two segments.
    - The "Past" segment represents embeddings that have already been input/output by/to the users.
    - The "Future" segment represents a linear sequence (not tree) of speculated future embeddings.
      + Initialized to embedding from a max-entropy probability distribution (anything is possible)
    - Note that we aren't speculating autoregressively. The speculation is computed in parallel.
    - ![probabilisitc speculation](assets/probabilistic_speculation.png)
1. Rather than the Transformer Blocks predicting the next token, they are simply auto-encoding
    - Like an encoder-style Masked Language Model (MLM), only we're going to generate text autoregressively from it.
    - It learns to minimize the entropy (uncertainty) in the input tokens
    - It's trained via disembedding on the zero-entropy one-hot encodings rather than explicitly minimizing entropy
    - The output is the same as input for past tokens, but for future tokens it squeezes as much entropy out as it can, giving a sequence that represents the probability distributions of the future timesteps.
1. We don't have to use a causal mask, however it is compatible with one to limit speculation distance.
    - it is desirable that past and future embeddings attend to each other, we can alway pad with more/less future embeddings depending on runtime conditions
    - it is expected that the entropy of the future embeddings will grow the further you speculate
        - this 'butterfly effect' is reflected in the uncertainty, eventually landing at max-entropy
    - tbd, if it's better to 'pin' past tokens to their zero-entropy output vs. allowing the past to drift (be recontextualized) as future embeddings collapse.
1. Transformer Blocks are actually computing a delta, which gets added back to the input iteratively
    - Similar to an iterative physics solver
    - This bares some resemblance to stable diffusion as well
    - A recent paper about Loop Transformers did this with a single block transformer, however I'm imagining it to be more than one.
    - Since we switched to an entropy-minimizing autoencoder, it means it's easy for an iterative solver to detect when it's converged (zero velocity/change) and early exit.  Now the cost can be proportional to the complexity of the problem, or even the available compute / latency requirements.


Rather than trying to speculate by computing a tree of possible definite futures, which is very expensive, we are going to use a single list of uncertain speculated embeddings.  You can think of these as the average of each layer of the tree, however we are going to directly predict them.  This will hopefully allow the attention to work across past and future, while preserving the uncertain nature of future events.

The outputs of the Transformer Blocks are initially handled similar to the decoder-LLM, during training loss is injected into each output via the disembedding projection, during inference things are a little different.

1. Rather than immediately projecting into Probability Space, we want to "collapse" one or more uncertain future embeddings into a certain (zero-entropy) embedding.
    - This allows us to autoregress in embedding space
    - The collapse of uncertainty about the future is how the past is minted in this system.  The collapsed embedding becomes part of the past segment in the context for the next autoregressive step.
    - I'll explain more of the details of this block after the analogy below.
1. We then convert this collapsed embedding into a token that can be output as text
    - This is only for the benefit of a human user, the output token/text is not used by the system
1. The collapsed embedding replaces the uncertain embedding
    - Then we continue reducing entropy like before and the rest of the context will respond to this collapse sort of like a vibration in a string
    - Autoregression is basically the same 'loop' we only take a slight detour to collapse, output, replace
    - Note that we're NOT starting from scratch each time we autoregress, the whole context window,including the speculated future tokens are already nearly-converged with minimal entropy, and that carries over to the next cycle losslessly.
1. We don't have to collapse JUST the first future embedding
    - we can explore strategies like collapsing embeddings below some entropy threshold
    - we can even collapse (or partially collapse) future embeddings non-causally. The other embeddings will adjust to fill in the gaps.
    - by continually partially collapsing the median-entropy embedding we may be able to 'draw out' the speculation horizon to some likely distant future as far into the future as needed for reasoning.
    - Note that we don't collapse all of the future tokens in one go, that would produce mostly gibberish because of the inherent uncertainty, however that doesn't mean that future embeddings aren't contributing useful information to the next embedding


![wcf diagram](assets/wfc_spaces.png)

The encoding spaces remain the same, only we can see the autoregression is done in embedding space via "Collapse" and the "Next Prediction" is replaced with "Entropy Reduction" (still a Transformer).


Let's now take a look at our toy-example embedding space and see with the context window looks like for WFC:
![wcf diagram](assets/wfc_autoencoding.png)

* This is the output of the Entropy Reduction step for the same example input (ABC)
* Note that the path contains multiple gray arrows, this is the speculated 'future' segment
  - they converge towards a gray pixel area in embedding space (high entropy superposition of tokens)
  - we don't collapse all of these arrows, just the first one (in this case) because we want to output it.
* The blue arrows indicate the 'collapse'.
  - Note that they aren't collapsing to the same points as the token embeddings (A, B, C), but rather to the nearest point that decodes to the same distribution (within a threshold)
  - This is intended to preserve some nuance in embedding space, while still collapsing to a determinate token
* The black arrows also don't have to go to target the token points exactly.  
  - Anywhere in the one-hot region will output the same token
  - Iterating in embedding space may give the network more freedom to store information in the different points that map to that token.
  - For example, look at the start of the path near A.  Initially that point started at A, however the network was better able to minimize entropy by drifting to its present location (still decodes to A)

## Quantum Mechanics Analogy

The analogy here isn't meant to be too literal, but I find it useful to explicitly state correspondences because doing so often leads to new ideas/insights.

| LLM Concept                    | Quantum Mechanics Concept     |  Why? |
|--------------------------------|-------------------------------|--------
| Token                          | Quantum State                 | discrete values that can be observed
| Logits / Probs                 | Superposition of States       | Uncertainty between one or more states
| Embedding / Context Window     | Wave Function                 | How the uncertainty is encoded
| Sampling                       | Measurement / Collapse        | The transition from uncertain -> certain mints the past
| Attention                      | Entanglement / Coherence      | Interdependencies between multiple particles
| Softmax                        | Uncertainty Principle         | The more certain you are of one state, the less you are of another
| Context Window                 | Spacetime                     | Context can represent temporal and/or spatial sequences depending on modality
| Speculation Tree               | Many Worlds Interpretation    | As you speculate, the number of possible discrete future timelines expands rapidly

From this, you can see why I call this idea 'Wave Function Collapse', the goal of the network is to minimize the entropy in the full spacetime wavefunction.  The future embeddings represent an ensemble probability distribution over all possible futures at each layer of a speculation tree.  Once this has converged, we can advance time by taking a 'measurement' of one of the next embedding, which collapses it from a superposition of possible states into a specific state (token).  Since all of the embeddings are entangled via attention, the next embedding collapse will cause changes to the remainder of the context/wave function (possibly non-causally), which pushes the uncertainty to the right (when we shift the context left, we fill in max-entropy embeddings on the right).


If you were to graph the entropy (uncertainty) of the context window, then the past segment on the left side of the context would be near zero entropy, and the future segment would ramp up towards the maximum entropy.

This is also true for a decoder-only LLM, however they typically only predict the next embedding (one in the future) at a time, so the entropy often doesn't get very high.


![decoder entropy](assets/decoder_entropy.png)

![wfc entropy](assets/wfc_entropy.png)

Notice that we're giving up some of the context window for the speculated future, however doing so may help with the accuracy of the next token (it has some think-ahead information to attend to).  In practice the future embeddings  would probably be a small fraction of the context, and most of the time the context isn't full anyways.  In either case, the number of future embeddings can be dynamically handled based on the shape of the entropy curve.  In the case where it's only 1 embedding, the graphs above would both look like the Decoder-Only LLM.

![wfc collapse](assets/wfc_collapse.png)

When we collapse a future embedding, we're sort of forcing a point on this graph down to zero.  The rest of the graph then re-converges to minimal entropy, which will have the effect of pushing the uncertainty 'hockey stick' over to the right (future).  I think of this like squeezing toothpaste out of a tube, or the wave of time moving into the future.


![wfc partial](assets/wfc_partial.png)

Normally we would fully collapse the next (first future) embedding, but we can also collapse more future embeddings, or even partially collapse embeddings.  You could imagine several iterations of holding a partial collapse followed by converging on a new minimal entropy.  This could be a way to extend out the time horizon for speculation for some plausible future.

![wfc lagrangian](assets/wfc_lagrangian.png)

In quantum mechanics, there is a concept called Lagrangian Mechanics that relates to the principle of least action.  Reversing the analogy can lead to new ideas related to the LLM idea.  For example, what if instead of collapsing the next token, we use some policy model to collapse something in the middle of the future segment?  The entropy-minimizing transformer would converge towards a state that tries to fill in the tokens to get you to this state.  This could allow for a more efficient search over possible futures to help with reasoning.

![wfc multiple lagrangian](assets/wfc_multiple_lagrangian.png)

At this point you might be wondering, how do we know what to collapse?