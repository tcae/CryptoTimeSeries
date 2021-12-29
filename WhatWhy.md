# How can we make complex systems less complex?

Complex systems are usually systmes of subsystems. Tranaktions change these subsystems. Main problem is how to maintain such comlex system - not only on a object basis but also to understand why certain concepts were introduction.

In order to use concepts at hand we have to be clear what we want to achieve but we must be able to extend a basic framework without loosing control due to complexity. Let's use Julia and see how far we come. Basically we have object concepts that can be reprresented as classes and transactions that can be represented as methods. The why and what can be covered with mandatory docstrings with the advantage that they are close to the implementation.

As this Julia project is about stock tradidng let's take a step back to put the project into the context of financial transactions. One basic class shall be value, which can be cryptocurencies, stocks, options, cash, ... but also other property, e.g. machines or houses, are value objects in that context.
Certain transactions can be applied to such value:

- sell or buy, i.e. tansform one value of class x to another value object of class y, which may require much more than just the transaction
- periodically or transactionally pay tax or fee for that value, which amy be different depending on the type of value owner organiztion, their legal residence or the residence of the value
- different approvals for certain transactions may be required, not only from the owner organization but also form external organizations like SEC or other governmental agencies
- it may be the case that certain agencies or individuals need to be informed, e.g. tax agencies, stakeholder individuals or organizations

The sell or buy transaction may seem as one transaction, e.g. form the view of a specific approver but it may break down in a series of transactions for the whole (prepare and sign legal contract, transport good) or parts of it (transfer different value objects from differnet accounts of the owner to the receiving accounts; break down the value object into transportable sub-objects, break down the object into sub-objects of different type that are individually assessed by customs because different approvals are required for the different types).