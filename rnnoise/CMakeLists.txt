cmake_minimum_required(VERSION 3.10)
project(rnnoise VERSION 1.0 LANGUAGES C)

# 🚨 VITAL: Fuerza la exportación de todos los símbolos públicos en Windows
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include_directories(include src)
file(GLOB SOURCES "src/*.c")

# Creamos la librería dinámica
add_library(rnnoise SHARED ${SOURCES})