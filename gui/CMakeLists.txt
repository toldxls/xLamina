set(CMAKE_AUTOMOC ON)
qt_add_executable(xlamina WIN32
    main.cpp
    mainwindow.cpp mainwindow.h
    imageview.cpp imageview.h
    worker.cpp worker.h)
target_link_libraries(xlamina PRIVATE stackcore Qt6::Widgets)
target_include_directories(xlamina PRIVATE ${CMAKE_SOURCE_DIR}/cli)
set_target_properties(xlamina PROPERTIES OUTPUT_NAME "xLamina")
install(TARGETS xlamina RUNTIME DESTINATION .)
