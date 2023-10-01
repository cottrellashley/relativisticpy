#ifndef NODE_PROVIDER_H
#define NODE_PROVIDER_H

typedef struct node {
    int item;
    struct node* left;
    struct node* right;
} Node;

// We're assuming we have a dynamic array-like structure called NodeList for storing the nodes
typedef struct {
    Node** nodes;
    size_t size;
    size_t capacity;
} NodeList;

// Bellow must be implemented !!

NodeList* create_node_list(size_t initial_capacity);
void append_to_node_list(NodeList* list, Node* node);
void free_node_list(NodeList* list);

Node* create(int value);
Node* insertRight(Node* root, int value);
Node* insertLeft(Node* root, int value);
Node* new_node(Node* left, Node* right);

 #endif