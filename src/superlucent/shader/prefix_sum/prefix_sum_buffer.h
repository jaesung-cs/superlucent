layout (binding = 0) buffer PrefixSumSsbo
{
  int size;
  int data[];
} prefix_sum;

layout (push_constant) uniform Query
{
  int level;
} query;
