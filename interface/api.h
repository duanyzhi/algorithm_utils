#ifndef ALU_API_H_
#define ALU_API_H_


// #if defined _WIN32 || defined __CYGWIN__
//   #ifdef BUILDING_DLL
//     #ifdef __GNUC__
//     #define ALU_API __attribute__ ((dllexport))
//     #else
//     #define ALU_API __declspec(dllexport)
//   #endif
// 
//   #else  // BUILDING_DLL
//     #ifdef __GNUC__
//     #define ALU_API __attribute__ ((dllimport))
//     #else
//     #define ALU_API __declspec(dllimport)
//     #endif
//   #endif
// 
// #define DLL_LOCAL
// #else
//   #if __GNUC__ >= 4
//     #define ALU_API __attribute__ ((visibility ("default")))
//     #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
//   #else
//     #define ALU_API
//     #define DLL_LOCAL
//   #endif
// #endif
// 

#define ALU_API __attribute__ ((visibility ("default")))

#endif  // ALU_API_H_

