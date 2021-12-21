# nn_final

##Introduction
    For our final project in CS342, we were tasked with programming a machine learning model that could play ice hockey in the SuperTuxKart racing game. Our model needed to be able to win against various pre-written agents in 2v2 tournaments where points are earned by scoring the hockey puck into the other team’s goal. After the game’s time limit, the winner is the team with the most goals scored. We elected to work on this project by building out an image-based agent because we were more comfortable tuning the vision component of our agent. 
##Model
    Our model is a fully convolutional network with 32 channels, 64 channels, and 128 channels along with up-convolutional channels in reverse channel size. We gathered our starting data for training the model by running the given agents against each other. This method gave us good results in the initial training, but when we wrote our controller, it struggled, and there were a lot of edge cases that we were having trouble with solving. To combat this, we did some dagger-style runs where we collected data from one run and trained a new model using it while getting rid of the older data. We continued this process 12 times before we thought our model was good enough. We further tried to improve our model by testing out a few different loss functions, and we saw the best results when using a binary cross-entropy loss function. 
##Controller
For our controller, we decided to go with a hand-tuned one rather than one of the provided ones. Our controller detects the puck and estimates the location of its center point. To do this, we built off of previous code from the third homework to try and segment out the puck from the current image. After segmenting the puck, we utilized a similar peak detection process to that from the fifth homework.
 If the puck is on the screen, then we treat it similarly to the aim point from the fifth homework assignment. Once the puck has been located, we determine the angle to the other team’s goal by taking the dot product and the arccosine. This math is easy to do because we have the location of both our players, and the enemy goal doesn’t move. In the case that the angle is very large or very small, our controller aims at the center of the puck. Otherwise, we try to offset the aim by hitting slightly to the left or right of the puck depending on the angle. The amount that we hit the puck by is determined by an inverse function of the distance to the goal because as the players get closer to the goal, the angle becomes more extreme. After a bit of testing, we realized that we were able to get similar results to this by hard-coding a constant amount left or right when we are within a small enough threshold distance from the enemy goal. 
When the puck is not on the screen, the controller looks to see if it has seen the puck recently. In this case that it has, the controller assumes that the puck is in the last location that it was seen at and plays accordingly. When the controller hasn’t seen the puck recently, it enters a “lost mode” where the player reverses backward towards their own goal to try and widen the amount of the map that they can see. To do this, we calculate the angle that we have to travel in to get to our goal and go backward in that direction. One problem that we noticed with this is that once we reached our goal, the puck could be too far away to see. While in this “lost mode,” our player kind of became a goalie in that it stayed around the goal. To fix this, we added a timer for the time we spent near the goal without seeing the puck. After the timer ran out, our player would travel towards the opponent's goal looking for the puck.
##Experimentation
    While coming up with a solution, we tried experimenting with a few different strategies. 
One example of this is that we messed around with choosing different karts for our members. While testing them out in different trial runs, we found that choosing Wilber increased the probability of hitting the puck with our strategy because of his bigger hitbox. After trying different combinations, we concluded that having both our players use the Wilber kart led to the best results.
 Another thing we tried was having the two players on our team perform different functions. We realized that our “lost mode” made the cars resemble goalies, and we thought that it would be interesting to see how our team would perform if we made one of the players act as a goalie for the entire game. To do this, we messed with the lost mode to make the goalie car immediately go back to its goal. This goalie car would then stay within a small range of the goal for the rest of the game trying to block the enemy cars from shooting the puck into its goal. The strategy seemed to work, but after some analysis, we found out that having both of our players on the offensive for the entire game led to the number of goals our team scored being much higher. We attempted to optimize the goalie strategy a little bit more before completely disregarding it by trying different karts for both positions. Surprisingly, having the larger Kart, Wilber, still seemed to outperform strategies with a mix of kart types, but the combination of Wilber as goalie and Tux as the offensive player was close. Because of these findings, we ditched the one-goalie strategy for the double offense one.
 Some additional things that we experimented with were using parameter search for the various player hyperparameters like target speed and drift threshold. This ended up giving us pretty suboptimal results, so we switched to manually tuning the values until we were satisfied with the performance. Something else that we tried was calculating the distance between a player and the puck by analyzing the size of the puck on the screen; the closer the puck, the larger the number of pixels it was. We did this by using the rectangular nature of the puck to make a bounding box with the maximum and minimum dimensions of the puck. This turned out to not be super useful, so we scrapped it in our final implementation.
