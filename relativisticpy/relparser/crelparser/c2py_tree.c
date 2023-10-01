#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int item;
    struct node* left;
    struct node* right;
} Node;

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

// Exported function to create sample tree
Node* create_sample_tree() {
    Node* root = create(1);
    insertLeft(root, 4);
    insertRight(root, 6);
    insertLeft(root->left, 42);
    insertRight(root->left, 3);
    insertLeft(root->right, 2);
    insertRight(root->right, 33);
    return root;
}
