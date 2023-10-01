#include "node_provider.h"

Node* create(int value) {
    Node* newNode = malloc(sizeof(Node));
    newNode->item = value;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

Node* insertLeft(Node* root, int value) {
    root->left = create(value);
    return root->left;
}

Node* insertRight(Node* root, int value) {
    root->right = create(value);
    return root->right;
}

NodeList* create_node_list(size_t initial_capacity) {
    NodeList* list = malloc(sizeof(NodeList));
    list->nodes = malloc(sizeof(Node*) * initial_capacity);
    list->size = 0;
    list->capacity = initial_capacity;
    return list;
}

void append_to_node_list(NodeList* list, Node* node) {
    if (list->size == list->capacity) {
        list->capacity *= 2;
        list->nodes = realloc(list->nodes, sizeof(Node*) * list->capacity);
    }
    list->nodes[list->size++] = node;
}