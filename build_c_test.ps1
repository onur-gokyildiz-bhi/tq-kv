# Setup MSVC environment
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
cmd /c "`"$vsPath`" x64 && cl /I tq-kv\include tq-kv\examples\test_ffi.c /Fe:target\release\test_ffi.exe /link /LIBPATH:target\release tq_kv.lib ws2_32.lib advapi32.lib userenv.lib bcrypt.lib ntdll.lib"
