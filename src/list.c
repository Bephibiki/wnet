#include "list.h"
#include <string.h>
#include <stdlib.h>

list *make_list(){
  list *l = malloc(sizeof(list));
  l->size = 0;
  l->front = 0;
  l->back = 0;
  return l;
}

int list_find(list *l, void *val){
  node * n= l->front;
  int i;
  while(n){
    if(n->val==val){
      return i;
    }
    n=n->next;
    ++i;
  }
  return -1;
}

void *list_pop(list *l){
  if(!l->back) return 0;
  node *b = l->back;
  void *val = b->val;
  l->back = b->prev;
  if(l->back) l->back->next = 0;
  free(b);
  --l->size;
  return val;
}

void list_insert(list *l, void * val){
  node * new = malloc(sizeof(node));
  new->val = val;
  new->next = 0;
  if(!l->back) {
    l->front = new;
    new->prev =0;
  } else {
    l->back->next = new;
    new->prev = l->back;
  }
  l->back = new;
  ++l->size;
}

void free_node(node *n){
  node *next;
  while(n){
    next = n->next;
    free(n);
    n=next;
  }
}

void **list_to_array(list *l){
  void **ary = calloc(l->size, sizeof(void*));
  int count =0;
  node *n = l->front;
  while(n){
    ary[count++] = n->val;
    n = n->next;
  }
  return ary;
}

void free_list(list *l){
  free_node(l->front);
  free(l);
}

