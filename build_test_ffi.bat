@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >/dev/null 2>&1
cd /d C:\Users\onurg\turkce-yazi-araci
cl /I tq-kv\include tq-kv\examples\test_ffi.c /Fe:target\release\test_ffi.exe /link /LIBPATH:target\release tq_kv.lib ws2_32.lib advapi32.lib userenv.lib bcrypt.lib ntdll.lib
