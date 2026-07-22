use easy_smt::{ContextBuilder, Response};

fn main() -> std::io::Result<()> {
    let mut ctx = ContextBuilder::new()
        .solver("z3")
        .solver_args(["-smt2", "-in"])
        .build()?;

    let int = ctx.int_sort();
    let pos = ctx.declare_const("pos", int)?;
    let num_threads = ctx.declare_const("num_threads", int)?;
    let x_len = ctx.declare_const("x_len", int)?;
    let y_len = ctx.declare_const("y_len", int)?;

    // assumes(...) + guard
    ctx.assert(ctx.gte(pos, ctx.numeral(0)))?;
    ctx.assert(ctx.lt(pos, num_threads))?;
    ctx.assert(ctx.eq(x_len, y_len))?;
    ctx.assert(ctx.gte(num_threads, y_len))?;
    ctx.assert(ctx.lt(pos, y_len))?; // guard taken

    // Negate the safety obligation: pos < x_len
    ctx.assert(ctx.not(ctx.lt(pos, x_len)))?;

    match ctx.check()? {
        Response::Unsat => println!("UNSAT (as expected): guard implies pos < x_len is proven"),
        Response::Sat => println!("SAT: counterexample exists -- property does NOT hold"),
        Response::Unknown => println!("UNKNOWN"),
    }
    Ok(())
}
