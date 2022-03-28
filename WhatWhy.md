# Strictly follow and why what argumentation

Complex systems are usually systmes of subsystems - a crypto trading system is no exception. Transactions change these subsystems. Main problem is how to maintain such comlex system - not only on a object basis but also to understand why certain concepts were introduction.

In order to use concepts at hand we have to be clear what we want to achieve but we must be able to extend a basic framework without loosing control due to complexity. Let's use Julia and see how far we come. Basically we have object concepts that can be reprresented as classes and transactions that can be represented as methods. The why and what can be covered with mandatory docstrings with the advantage that they are close to the implementation.

As this Julia project is about stock tradidng let's take a step back to put the project into the context of financial transactions. One basic class shall be value, which can be cryptocurencies, stocks, options, cash, ... but also other property, e.g. machines or houses, are value objects in that context.
Certain transactions can be applied to such value:

- sell or buy, i.e. tansform one value of class x to another value object of class y, which may require much more than just the transaction
- periodically or transactionally pay tax or fee for that value, which amy be different depending on the type of value owner organiztion, their legal residence or the residence of the value
- different approvals for certain transactions may be required, not only from the owner organization but also form external organizations like SEC or other governmental agencies
- it may be the case that certain agencies or individuals need to be informed, e.g. tax agencies, stakeholder individuals or organizations

The sell or buy transaction may seem as one transaction, e.g. form the view of a specific approver but it may break down in a series of transactions for the whole (prepare and sign legal contract, transport good) or parts of it (transfer different value objects from differnet accounts of the owner to the receiving accounts; break down the value object into transportable sub-objects, break down the object into sub-objects of different type that are individually assessed by customs because different approvals are required for the different types).

In crypto trading it is all about yield. We have the follwig main task:

- train/improve a trading system: observe historic data and dervie from that a trained system that does gain a positive yield with trading
- execute a trading system: assess current data and execute conclusions about trade actions by using the trained system

These are 2 independent loops. As soon as a better trading system is available it will replace the old.

# Train/improve a trading system

The minimal requirement to allow a trading system to execute is a positive yield on average. This requires self assessment and a definition when a system is good / not good enough and it is risk acceptable to trade or better to not trade at all. Both criteria should form a hysteresis to avoid high frequent on/off switches.

## Definition of fit for purpose

Evaluate the system on historic data to determine what is an acceptable deviation from positive yield and if it is good enough then eveluate the number of consequtive time window, e.g. a day, or consecutive losses before it falls out of the accepted range observed.

good = x consecutive days showing at least 1 day positive trade result AND the integrate loss of recent (without spec)
