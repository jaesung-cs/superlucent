#include <superlucent/application.h>

#include <iostream>
#include <stdexcept>

int main()
{
  supl::Application app;

  try
  {
    app.Run();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
