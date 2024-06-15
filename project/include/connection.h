#ifndef CONNECTION_H
#define CONNECTION_H

#include <iostream>
#include <vector>



class Connection{


    public:
        Connection();
        double weight;
        double deltaWeight;
    private:
        double randomWeight();
};


#endif