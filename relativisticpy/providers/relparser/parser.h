



struct node
{
    int data;
    struct node *left;
    struct node *right;
};


struct node* create(value) {
  struct node* newNode = malloc(sizeof(struct node));
  newNode->data = value;
  newNode->left = NULL;
  newNode->right = NULL;

  return newNode;
}

// Insert on the left of the node.
struct node* insertLeft(struct node* root, int value) {
  root->left = create(value);
  return root->left;
}

// Insert on the right of the node.
struct node* insertRight(struct node* root, int value) {
  root->right = create(value);
  return root->right;
}
