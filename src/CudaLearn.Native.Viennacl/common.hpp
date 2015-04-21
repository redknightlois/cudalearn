#pragma once

#ifndef VC_EXTRALEAN
	#define VC_EXTRALEAN
#endif


#ifdef _WINDOWS
#define DLLEXPORT __declspec( dllexport )
#else
#define DLLEXPORT
#endif

#include <viennacl\scalar.hpp>
#include <viennacl\vector.hpp>
#include <viennacl\vector_proxy.hpp>
#include <viennacl\matrix.hpp>
#include <viennacl\matrix_proxy.hpp>
#include <viennacl\linalg\inner_prod.hpp>
#include <viennacl\linalg\vector_operations.hpp>
#include <viennacl\linalg\matrix_operations.hpp>
#include <viennacl\linalg\norm_1.hpp>
#include <viennacl\linalg\prod.hpp>

