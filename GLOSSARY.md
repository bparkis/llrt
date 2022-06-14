# Glossary

Some of the more frequent terms used in LLRT.

* Component: a multidimensional array of nodes
* Node: a part of a Component, containing an arbitrary C++ object
* Link: a connection between the nodes of two Components, represented algorithmically as a pattern of iteration
* Edge: a connection between two nodes
* Network: a set of Components and Links
* Link end: the end of a Link closest to a particular component
* 0-end: the end of a Link with index 0
* 1-end: the end of a Link with index 1
* Edge end: the end of an edge closest to a particular node, containing an arbitrary C++ object.
* Kernel: a function applied to every edge in a Link from a particular end
* Self-link: the Link that every Component has, connecting each node in the Component to itself. Iteration over the self-link is iteration over the nodes in the Component.
* Near component: the component where we are
* Near node: the node where we are
* Far component: the component at the other end of the Link
* Far node: the node at the other end of the edge
* Axon: an arbitrarily designated end of a Link
* Dendrite: the end of the Link that is not the Axon
* LinkType: the object wrapped by a Link that gives the pattern of iteration for that Link
* Link operation: the process of applying a kernel to every edge of a Link from a particular end
* Network operation: a set of Link operations with the same kernel, produced by a single call to one of the Process* family of functions
* Operation: could mean a Link operation or a Network operation
* Kernel parameter specifier: a sequence of characters associated with an operation, specifying which parameters the kernel accepts, out of all the possible parameters.
* Scheduler: the module that manages worker threads for parallel execution
* Batch: a set of Link operations, possibly from multiple calls to the Process* family of functions, submitted to the Scheduler together as a unit. Operations in a Batch may be executed in any order or simultaneously.
* Job chunk: a partial execution of a Link operation, assigned to a single worker thread
* Near-node guarantee: when a worker thread runs a kernel on an edge with a particular near node, no other worker thread will run a kernel with that same near node at the same time.
